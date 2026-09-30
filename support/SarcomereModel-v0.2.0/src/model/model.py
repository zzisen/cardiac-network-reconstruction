# Copyright (c) 2025-2026 Daniel Härtter
# Licensed under the MIT License.

import os
import pickle
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Union

import numpy as np
from scipy.integrate import ode, solve_ivp
from scipy.ndimage import label

from .utils import get_random_id
from .analysis import *


@dataclass
class ModelParams:
    """Model parameters for sarcomere dynamics
    
    Attributes
    ----------
    N : int
        Number of sarcomeres
    mu : float
        Inertia
    eta : List[float]
        Fluctuation amplitudes [thermal, active]
    k_l : float
        Stiffness constant of load
    k_s : List[float]
        Stiffness constants of sarcomere [linear, quadratic]
    u_s : float
        Sarcomere-level friction [coefficient]
    o_s : Optional[List[float]]
        Length-dependent force parameters [x0, xmin, xmax] or None
    f_s : List[float]
        Force-velocity relation [slope, intercept, scale, sigmoid_sharpness]
    h : float
        Static force inhomogeneity (STD of normal dist)
    """
    N: int
    mu: float
    eta: List[float]
    k_l: float
    k_s: List[float]
    u_s: float
    o_s: Optional[List[float]] 
    f_s: List[float]
    h: float = 0.0
    
    def save_pickle(self, filepath: str):
        """Save parameters to pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_json(self, filepath: str):
        """Save parameters to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load_pickle(cls, filepath: str) -> 'ModelParams':
        """Load parameters from pickle file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'ModelParams':
        """Load parameters from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class SimParams:
    """Simulation parameters
    
    Attributes
    ----------
    t1 : float
        Duration of simulation
    dt : float
        Simulation time step
    a_mode : str
        Activation mode ('sine' or 'modified_sine')
    a : List[float]
        Activation parameters
    p : float
        Pre-strain
    y0 : Optional[np.ndarray]
        Initial conditions (ndarray or None for automatic)
    """
    t1: float
    dt: float
    a_mode: str
    a: List[float]
    p: float
    y0: Optional[np.ndarray] = None
    
    def save_pickle(self, filepath: str):
        """Save parameters to pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def save_json(self, filepath: str):
        """Save parameters to JSON file (excluding y0 array)"""
        data = asdict(self)
        if self.y0 is not None:
            data['y0'] = self.y0.tolist()
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_pickle(cls, filepath: str) -> 'SimParams':
        """Load parameters from pickle file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'SimParams':
        """Load parameters from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        if data['y0'] is not None:
            data['y0'] = np.array(data['y0'])
        return cls(**data)
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)


class Model:
    """Dynamic model to simulate collective behavior of coupled sarcomeres in cardiac myofibril

    Parameters
    ----------
    model_params : ModelParams or dict
        Model parameters (can be ModelParams dataclass or dict for backward compatibility)
    sim_params : SimParams or dict
        Simulation parameters (can be SimParams dataclass or dict for backward compatibility)
    folder : str, optional
        Folder to store results
    autosave : bool, optional
        Whether to enable autosave. Default is False.
    verbose : bool, optional
        Whether to print diagnostic messages. Default is True.
    """

    def __init__(self, 
                 model_params: Union[ModelParams, dict], 
                 sim_params: Union[SimParams, dict], 
                 folder: Optional[str] = None, 
                 autosave: bool = False,
                 verbose: bool = True):
        """Initialize model"""
        # Convert dicts to dataclasses for backward compatibility
        if isinstance(model_params, dict):
            self.model_params = ModelParams(**model_params)
        else:
            self.model_params = model_params
            
        if isinstance(sim_params, dict):
            self.sim_params = SimParams(**sim_params)
        else:
            self.sim_params = sim_params
        
        self.autosave = autosave
        self.verbose = verbose
        self.name = get_random_id(8)
        
        if autosave:
            if folder is None:
                raise ValueError("folder must be specified when autosave is True.")
            self.model_folder = folder + '/' + self.name + '/'
            os.makedirs(self.model_folder, exist_ok=True)
        else:
            self.model_folder = None
        
        # dict for results
        self.data = {}
        
        # define random term eta and time arrays
        self.num_steps = int(self.sim_params.t1 / self.sim_params.dt)
        self.eta_thermal = np.random.normal(
            0, self.model_params.eta[0],
            size=(self.model_params.N, self.num_steps)
        ) / np.sqrt(self.sim_params.dt)
        self.eta_active = np.random.normal(
            0, self.model_params.eta[1],
            size=(self.model_params.N, self.num_steps)
        ) / np.sqrt(self.sim_params.dt)
        
        # initialize arrays
        self.n = 0
        self.y = np.zeros((self.num_steps, 2 * self.model_params.N)) * np.nan
        if self.sim_params.y0 is None:
            # Zero initial velocities, small position perturbations
            velocities = np.zeros(self.model_params.N)
            positions = np.random.normal(0, 0.02, self.model_params.N)
            self.sim_params.y0 = np.concatenate([velocities, positions])

        self.time = np.zeros(self.num_steps)
        
        # define activation
        self.time_act = np.linspace(0, self.sim_params.t1, self.num_steps, endpoint=True)
        if self.sim_params.a_mode == 'sine':
            self.act = np.sin(
                self.time_act * self.sim_params.a[0] * 2 * np.pi
            ) * self.sim_params.a[2]
            self.act = self.act.clip(min=self.sim_params.a[1])
        elif self.sim_params.a_mode == 'modified_sine':
            assert len(self.sim_params.a) == 4, (
                "Please enter all parameters for 'modified_sine' mode: "
                "(frequency, baseline, amplitude, time_contr)"
            )
            assert self.sim_params.a[3] < 1/self.sim_params.a[0], "time_contr too long for frequency."
            self.act = np.ones_like(self.time_act) * self.sim_params.a[1]
            start_contr = np.arange(0, self.sim_params.t1, 1/self.sim_params.a[0])
            time_sine = np.arange(0, self.sim_params.a[3], self.sim_params.dt)
            len_time_sin = len(time_sine)
            sine = (
                np.sin(time_sine / self.sim_params.a[3] * np.pi) * self.sim_params.a[2]
            ) + self.sim_params.a[1]
            clipped_sine = np.clip(sine, self.sim_params.a[1], None)
            for contr in start_contr:
                _contr = int(contr / self.sim_params.dt)
                self.act[_contr: min(_contr + len_time_sin, len(self.act))] = clipped_sine[
                    : min(len_time_sin, len(self.act) - _contr)
                ]
        else:
            raise ValueError(self.sim_params.a_mode + ' not valid.')
        
        # sarcomere force inhomogeneity
        if self.model_params.h == 0:
            self.h = 1
        else:
            self.h = np.random.normal(1, self.model_params.h, self.model_params.N)

    def _print(self, message: str, **kwargs):
        """
        Print message only if verbose is enabled.
        
        Parameters
        ----------
        message : str
            Message to print
        **kwargs
            Additional arguments passed to print()
        """
        if self.verbose:
            print(message, **kwargs)

    def n_active_motors(self, v):
        """Calculate fraction of active motors."""
        return 1 / (1 + np.exp(-v * self.model_params.f_s[3]))

    def force_vel_rel_poly(self, v):
        """Polynomial force-velocity relation."""
        f_s = self.model_params.f_s
        return f_s[0] + f_s[1] * v + f_s[2] * v ** 2 + f_s[3] * v ** 3

    def force_vel_rel_sarc(self, v, x, h, act, model='poly'):
        """Calculate sarcomere force-velocity relation."""
        if model == 'poly':
            force_vel_rel = self.force_vel_rel_poly
        else:
            raise ValueError(f'model {model} not implemented.')
        return force_vel_rel(v) * self.overlap(x) * h * act + self.model_params.u_s * v

    def overlap(self, x):
        """Calculate filament overlap function."""
        if self.model_params.o_s is None:
            return np.ones_like(x)
        elif isinstance(self.model_params.o_s, list):
            x0, xmin, xmax = self.model_params.o_s
            return np.piecewise(
                x, 
                [(x < xmin), (x >= xmin) & (x <= x0), (x > x0) & (x < xmax), (x >= xmax)],
                [0, lambda z: (z - xmin) / (x0 - xmin), lambda z: -(z - xmax) / (xmax - x0), 0]
            )

    def force_single(self, x, v, act):
        """Calculate single sarcomere force."""
        k_s = self.model_params.k_s
        f_s = (
            - k_s[0] * x * np.heaviside(x, 1) 
            - (k_s[0] * x + k_s[1] * x ** 2) * np.heaviside(-x, 1)
        )
        return 1 / self.model_params.mu * (f_s - self.force_vel_rel_sarc(v, x, self.h, act))

    def force_coupling(self, x):
        """Calculate coupling force between sarcomeres."""
        return (
            - 1 / self.model_params.mu 
            * self.model_params.k_l / self.model_params.N 
            * (np.sum(x) - self.sim_params.p)
        )

    def integrate(self):
        """
        Uses Euler-Maruyama method suitable for SDEs.
        """
        # Validate parameters before starting
        if self.model_params.mu <= 0:
            raise ValueError(f"Invalid mu: {self.model_params.mu}")
        if self.sim_params.dt <= 0:
            raise ValueError(f"Invalid dt: {self.sim_params.dt}")

        # Initialize
        y = self.sim_params.y0.copy()
        
        for self.n in range(self.num_steps):
            t = self.n * self.sim_params.dt
            self.time[self.n] = t
            
            # Get velocities and positions
            v = y[:self.model_params.N]
            x = y[self.model_params.N: 2 * self.model_params.N]
            
            # Deterministic forces
            f_det = self.force_single(x, v, self.act[self.n]) + self.force_coupling(x)
            
            # Stochastic forces
            eta = (
                self.eta_thermal[:, self.n]
                + self.eta_active[:, self.n] * self.n_active_motors(v) * self.act[self.n]
            )
            
            # Euler-Maruyama update
            v_new = v + (f_det + eta) * self.sim_params.dt
            x_new = x + v * self.sim_params.dt
            
            # Update state
            y = np.concatenate([v_new, x_new])
            self.y[self.n, :] = y
        
        # Unwrap results
        self.data['time'] = self.time
        self.data['act'] = self.act
        self.data['contr'] = self.data['act'] > self.sim_params.a[1]
        self.data['labels_contr'], self.data['n_contr'] = label(self.data['contr'])
        self.data['start_contr'] = np.asarray([
            np.argwhere(self.data['labels_contr'] == l)[0][0] 
            for l in np.arange(1, self.data['n_contr'] + 1)
        ])
        self.data['time_contr_tps'] = int(np.sum(self.data['contr']) / self.data['n_contr'])
        self.data['time_contr'] = self.data['time_contr_tps'] * self.sim_params.dt
        self.data['vel'] = -self.y.T[0:self.model_params.N]
        self.data['vel_avg'] = -np.nanmean(self.y.T[0:self.model_params.N], axis=0)
        self.data['length'] = -self.y.T[self.model_params.N:]
        self.data['length_avg'] = -np.nanmean(self.y.T[self.model_params.N:], axis=0)
        
        if self.autosave:
            self.save_model()


    def analyze(self, popping_threshold=0.25):
        """
        Analyze model results.
        
        Parameters
        ----------
        popping_threshold : float, optional
            Threshold for detecting popping events
        """
        analyze_trajs(self)
        analyze_popping(self, thres_popping=popping_threshold, plot=False)
        analyze_popping_tau_dist(self)
        correlation_cycles_mutual_serial(self)

    def save_model(self, format='pickle'):
        """
        Save model data and parameters.
        
        Parameters
        ----------
        format : str
            'pickle' or 'json' (default: 'pickle')
        """
        if format == 'pickle':
            with open(self.model_folder + 'data.p', 'wb') as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.model_params.save_pickle(self.model_folder + 'model_params.p')
            self.sim_params.save_pickle(self.model_folder + 'sim_params.p')
        elif format == 'json':
            with open(self.model_folder + 'data.p', 'wb') as handle:
                pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.model_params.save_json(self.model_folder + 'model_params.json')
            self.sim_params.save_json(self.model_folder + 'sim_params.json')
        else:
            raise ValueError(f"Format {format} not supported. Use 'pickle' or 'json'.")
    
    @classmethod
    def load_model(cls, folder, format='pickle', name=None):
        """
        Load a saved model.
        
        Parameters
        ----------
        folder : str
            Base folder containing saved models
        format : str
            'pickle' or 'json' (default: 'pickle')
        name : str, optional
            Model name (random ID). If None, uses the folder path directly.
            
        Returns
        -------
        Model
            Loaded model instance
        """
        if name is not None:
            model_folder = folder + '/' + name + '/'
        else:
            model_folder = folder if folder.endswith('/') else folder + '/'
        
        # Load parameters
        if format == 'pickle':
            model_params = ModelParams.load_pickle(model_folder + 'model_params.p')
            sim_params = SimParams.load_pickle(model_folder + 'sim_params.p')
        elif format == 'json':
            model_params = ModelParams.load_json(model_folder + 'model_params.json')
            sim_params = SimParams.load_json(model_folder + 'sim_params.json')
        else:
            raise ValueError(f"Format {format} not supported. Use 'pickle' or 'json'.")
        
        # Load data
        with open(model_folder + 'data.p', 'rb') as handle:
            data = pickle.load(handle)
        
        # Create model instance
        model = cls(model_params, sim_params, folder=folder, autosave=False)
        model.data = data
        if name is not None:
            model.name = name
        model.model_folder = model_folder
        
        return model

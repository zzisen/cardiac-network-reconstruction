# Copyright (c) 2025-2026 Daniel Härtter
# Licensed under the MIT License.

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib import cm
import matplotlib as mpl
from tqdm import tqdm
from typing import Optional, Tuple


# plot params
fontsize = 8
markersize = 3
labelpad = 1
dpi = 600
save_format = 'png'

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

width_1cols = 3.5
width_1p5cols = 5
width_2cols = 7.1
plt.rcParams.update({'font.size': fontsize, 'axes.labelpad': labelpad, 'font.family': 'arial'})


def label_all_panels(axs, offset=(-0.1, 1.1), color='k'):
    """
    Label all uppercase-keyed panels in a dictionary of axes.

    Parameters
    ----------
    axs : dict
        Dictionary of matplotlib Axes objects.
    offset : tuple, optional
        The (x, y) offset for label placement in axes coordinates.
    color : str, optional
        Color of the label text.
    """
    for key in axs.keys():
        if not key.isupper():
            continue
        label_panel(axs[key], key, offset=offset, color=color)


def label_panel(ax, label, offset=(-0.1, 1.1), color='k'):
    """
    Add a text label to a single panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to label.
    label : str
        The label text to display.
    offset : tuple, optional
        The (x, y) offset for label placement in axes coordinates.
    color : str, optional
        Color of the label text.
    """
    ax.text(offset[0], offset[1], label, transform=ax.transAxes,
            fontsize=fontsize + 1, fontweight='black', va='top', ha='right', color=color)


def remove_all_spines(axs):
    """
    Remove top and right spines from all axes in a dictionary.

    Parameters
    ----------
    axs : dict
        Dictionary of matplotlib Axes objects.
    """
    for key in axs.keys():
        remove_spines(axs[key])


def remove_spines(ax):
    """
    Remove top and right spines from a single axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    """
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def polish_xticks(ax, major, minor, pad=3):
    """
    Format x-axis tick marks with major and minor locators.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    major : float
        Spacing for major tick marks.
    minor : float
        Spacing for minor tick marks.
    pad : float, optional
        Padding between axis and tick labels in points.
    """
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.tick_params(axis='x', pad=pad)


def polish_yticks(ax, major, minor, pad=3):
    """
    Format y-axis tick marks with major and minor locators.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify.
    major : float
        Spacing for major tick marks.
    minor : float
        Spacing for minor tick marks.
    pad : float, optional
        Padding between axis and tick labels in points.
    """
    ax.yaxis.set_major_locator(MultipleLocator(major))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_minor_locator(MultipleLocator(minor))
    ax.tick_params(axis='y', pad=pad)


def plot_trajs(axs_list, model, tlim=None):
    """
    Plot length and velocity trajectories with activation overlay.

    Parameters
    ----------
    axs_list : list of matplotlib.axes.Axes
        List containing two axes: [length_ax, velocity_ax].
    model : Model
        Model instance containing simulation data.
    tlim : tuple or None, optional
        Time limits (tmin, tmax) for x-axis.
    """
    ax01 = plt.twinx(axs_list[0])
    axs_list[0].axhline(0, linewidth=1, c='k', linestyle=':')
    ax01.fill_between(model.time, 0, model.data['act'], color='y', lw=1, alpha=0.2, edgecolor=None)
    axs_list[0].plot(model.time, model.data['length'].T, linewidth=0.5)
    axs_list[0].plot(model.time, np.nanmean(model.data['length'], axis=0), linewidth=2, c='k', linestyle='--',
                     label='Avg.')
    axs_list[0].set_xlabel('Time', labelpad=labelpad)
    axs_list[0].set_ylabel('Length $x$', labelpad=labelpad)
    ax01.set_ylabel('Activation $a$')
    axs_list[0].set_xlim(tlim)
    ax01.set_ylim(0, None)
    axs_list[0].legend(fontsize=fontsize, loc=1)
    ax01.zorder = 0
    axs_list[0].zorder = 1
    axs_list[0].patch.set_visible(False)

    ax11 = plt.twinx(axs_list[1])
    axs_list[1].axhline(0, linewidth=1, c='k', linestyle=':')
    ax11.fill_between(model.time, 0, model.data['act'], color='y', lw=1, alpha=0.2, edgecolor=None)
    axs_list[1].plot(model.time, model.data['vel'].T, linewidth=0.5)
    axs_list[1].plot(model.time, np.nanmean(model.data['vel'], axis=0), linewidth=2, c='k', linestyle='--',
                     label='Avg.')
    axs_list[1].set_xlabel('Time', labelpad=labelpad)
    axs_list[1].set_ylabel('Velocity $v$', labelpad=labelpad)
    ax11.set_ylabel('Activation $a$')
    axs_list[1].set_xlim(tlim)
    ax11.set_ylim(0, None)
    axs_list[1].legend(fontsize=fontsize, loc=1)
    ax11.zorder = 0
    axs_list[1].zorder = 1
    axs_list[1].patch.set_visible(False)


def plot_vel_force(ax, model, xlim=None, ylim=None, loc=3):
    """
    Plot force-velocity relationship with trajectory overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    model : Model
        Model instance containing simulation data.
    xlim : tuple or None, optional
        Velocity axis limits (vmin, vmax).
    ylim : tuple or None, optional
        Force axis limits (fmin, fmax).
    loc : int, optional
        Legend location code.
    """
    # plot force-velocity relation
    v = np.linspace(np.nanmin(model.data['vel']) - 5, np.nanmax(model.data['vel']) + 5, 200)
    f = model.force_vel_rel_sarc(v, x=0, h=1, act=model.sim_params.a[2])

    # plot trajectories
    for i in range(model.model_params.N):
        ax.plot(model.data['vel'].T, model.data['f_p'].T, linewidth=0.5)
    if model.model_params.N > 1:
        ax.plot(model.data['vel_avg'], model.data['f_p_avg'], '-', linewidth=2, c='k')

    # plot f-v-relation
    fv_plot = ax.plot(-v, -f, '--', c='r', linewidth=2)

    ax.axhline(0, linewidth=1, c='k', linestyle='--')
    ax.axvline(0, linewidth=1, c='k', linestyle='--')
    ax.set_xlabel('Velocity $v$', labelpad=labelpad)
    ax.set_ylabel('Force', labelpad=labelpad)

    if xlim is None:
        ax.set_xlim(np.nanmax(model.data['vel']) + 1.5, np.nanmin(model.data['vel']) - 1.5)
    else:
        ax.set_xlim(xlim)
    if ylim is None:
        ax.set_ylim(np.nanmax(model.data['f_p']) + 0.5, np.nanmin(model.data['f_p']) - 1.2)
    else:
        ax.set_ylim(ylim)

    ax.legend(fv_plot, ['$F(v, a)$'], loc=loc, fontsize=fontsize)


def plot_vel_length(ax, model):
    """
    Plot velocity vs length phase space trajectory.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    model : Model
        Model instance containing simulation data.
    """
    # plot trajectories
    for i in range(model.model_params.N):
        ax.plot(model.data['length'].T, model.data['vel'].T, linewidth=0.5)
    if model.model_params.N > 1:
        ax.plot(model.data['length_avg'], model.data['vel_avg'], ':', linewidth=2, c='k')

    ax.axhline(0, linewidth=1, c='k', linestyle=':')
    ax.axvline(0, linewidth=1, c='k', linestyle=':')
    ax.set_ylabel('Velocity $v$', labelpad=labelpad)
    ax.set_xlabel('Length $x$', labelpad=labelpad)


def plot_force_vel_relation(ax, model, a=None, colorbar=True, vlim=(-3, 3), linewidth=1.5):
    """
    Plot force-velocity relation curves for different activation levels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    model : Model
        Model instance containing force-velocity relation.
    a : float or None, optional
        Single activation level to plot. If None, plots multiple levels.
    colorbar : bool, optional
        Whether to display colorbar for activation levels.
    vlim : tuple, optional
        Velocity range (vmin, vmax) for the plot.
    linewidth : float, optional
        Line width for the curves.
    """
    v = np.linspace(vlim[0], vlim[1], 100)
    n_steps = 10
    cmap = cm.get_cmap('viridis', n_steps)
    if a is None:
        for i, a in enumerate(np.linspace(0, 1, n_steps)):
            fvr = model.force_vel_rel_sarc(v, x=0, h=1, act=a)
            ax.plot(-v, fvr, c=cmap(i), lw=linewidth)
    else:
        fvr = model.force_vel_rel_sarc(v, x=0, h=1, act=a)
        ax.plot(-v, fvr, c='royalblue', label='F-V relation')
    ax.axhline(0, c='k', linestyle='--', lw=1)
    ax.axvline(0, c='k', linestyle='--', lw=1)
    if colorbar:
        plt.colorbar(cm.ScalarMappable(norm=None, cmap=cmap), ax=ax, label='Activation $c$')


def plot_model_summary(model, title=None, savename=None):
    """
    Create comprehensive summary plot with trajectories and phase space.

    Parameters
    ----------
    model : Model
        Model instance containing simulation data.
    title : str or None, optional
        Figure title.
    savename : str or None, optional
        File path to save the figure. If None, figure is not saved.
    """
    mosaic = """
    AC
    BC
    """

    fig, axs = plt.subplot_mosaic(mosaic, figsize=(width_2cols, width_1cols), facecolor='w', constrained_layout=True)

    # trajectories
    plot_trajs([axs['A'], axs['B']], model, tlim=(0, 2.7))

    # trajectory in phase space
    plot_phase_space(axs['C'], model)

    # force-velocity relation
    ax_inset = axs['C'].inset_axes([0.55, 0.55, 0.42, 0.42])
    plot_force_vel_relation(ax_inset, model, colorbar=False)
    fig.suptitle(title, fontsize=fontsize - 2)
    plt.tight_layout()
    label_all_panels(axs)
    if savename is not None:
        fig.savefig(savename, dpi=300)
    plt.show()


def plot_phase_space(ax, model, tlim=(0, 2.8), x_lim=None, y_lim=None):
    """
    Plot velocity vs length phase space diagram for all sarcomeres.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    model : Model
        Model instance containing simulation data.
    tlim : tuple, optional
        Time window (tmin, tmax) to display.
    x_lim : tuple or None, optional
        Velocity axis limits (vmin, vmax).
    y_lim : tuple or None, optional
        Length axis limits (lmin, lmax).
    """
    # get data
    delta_slen = model.data['length']
    vel = model.data['vel']
    delta_slen_avg = model.data['length_avg']
    vel_avg = model.data['vel_avg']
    # colormap
    cm = plt.cm.nipy_spectral(np.linspace(0, 1, model.model_params.N))
    ax.set_prop_cycle('color', list(cm))
    # plot limits and params
    idxlim = (int(tlim[0] / model.sim_params.dt), int(tlim[1] / model.sim_params.dt))
    for vel_i, delta_i in zip(vel, delta_slen):
        ax.plot(vel_i[idxlim[0]:idxlim[1]], delta_i[idxlim[0]:idxlim[1]], c='darkgrey', lw=0.36, alpha=0.8)

    ax.plot(vel_avg[idxlim[0]:idxlim[1]], delta_slen_avg[idxlim[0]:idxlim[1]], c='k', lw=2)

    ax.set_xlabel('Velocity $v$')
    ax.set_ylabel('Length $x$')

    ax.axhline(0, lw=1, c='k', linestyle='--', alpha=0.4)
    ax.axvline(0, lw=1, c='k', linestyle='--', alpha=0.4)

    polish_xticks(ax, 4, 2)
    polish_yticks(ax, 0.2, 0.1)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def plot_overlay_length(ax, model, tlim=(0, 2.8), y_lim=None):
    """
    Plot sarcomere length trajectories with activation overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    model : Model
        Model instance containing simulation data.
    tlim : tuple, optional
        Time window (tmin, tmax) to display.
    y_lim : tuple or None, optional
        Length axis limits (lmin, lmax).
    """
    # plot limits and params
    idxlim = (int(tlim[0] / model.sim_params.dt), int(tlim[1] / model.sim_params.dt))
    # plot single and average trajectories
    ax01 = plt.twinx(ax)
    ax.axhline(0, linewidth=1, c='k', linestyle=':')
    ax01.fill_between(model.time, model.data['act'], color='lavender', lw=1, edgecolor=None)
    # colormap
    cm = plt.cm.nipy_spectral(np.linspace(0, 1, model.model_params.N))
    ax.set_prop_cycle('color', list(cm))
    ax.plot(model.time, model.data['length'].T, linewidth=0.6)
    ax.plot(model.time, np.nanmean(model.data['length'], axis=0), linewidth=2, c='k', linestyle='-',
            label='Avg.')
    ax.set_xlabel('Time', labelpad=labelpad)
    ax.set_ylabel('Length $x$', labelpad=labelpad)
    ax01.set_yticks([])
    ax.set_xlim(tlim)
    ax.set_ylim(y_lim)
    ax01.zorder = 0
    ax.zorder = 1
    ax.patch.set_visible(False)

    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))



def plot_overlay_velocity(ax, model, tlim=(0, 2.8), y_lim=(-4, 7), show_contr=True):
    """
    Plot sarcomere velocity trajectories with activation overlay.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to plot on.
    model : Model
        Model instance containing simulation data.
    tlim : tuple, optional
        Time window (tmin, tmax) to display.
    y_lim : tuple, optional
        Velocity axis limits (vmin, vmax).
    show_contr : bool, optional
        Whether to show contraction overlay (currently unused).
    """
    # plot limits and params
    idxlim = (int(tlim[0] / model.sim_params.dt), int(tlim[1] / model.sim_params.dt))
    # plot single and average trajectories
    ax01 = plt.twinx(ax)
    ax.axhline(0, linewidth=1, c='k', linestyle=':')
    ax01.fill_between(model.time, model.data['act'], color='k', lw=1, alpha=0.1, edgecolor=None)
    # colormap
    cm = plt.cm.nipy_spectral(np.linspace(0, 1, model.model_params.N))
    ax.set_prop_cycle('color', list(cm))
    ax.plot(model.time, model.data['vel'].T, linewidth=0.6)
    ax.plot(model.time, np.nanmean(model.data['vel'], axis=0), linewidth=2, c='k', linestyle='-',
            label='Avg.')
    ax.set_xlabel('Time', labelpad=labelpad)
    ax.set_ylabel('Velocity $v$', labelpad=labelpad)
    ax01.set_yticks([])
    ax.set_xlim(tlim)
    ax.set_ylim(y_lim)
    ax01.zorder = 0
    ax.zorder = 1
    ax.patch.set_visible(False)

    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_minor_locator(MultipleLocator(0.25))


def animate_force_velocity(
    model,
    forces: np.ndarray,
    output_path: str,
    tlim: Tuple[float, float] = (0, 5),
    ylim: Optional[Tuple[float, float]] = None,
    vlim: Tuple[float, float] = (-4.5, 9),
    force_ylim: Tuple[float, float] = (-1.5, 1.5),
    figsize: Tuple[float, float] = (2.8, 3.5),
    dpi: int = 300,
    fps: int = 50,
    interval: int = 20,
    colormap: str = 'nipy_spectral',
    preset: str = 'fast',
    bitrate: int = 5000,
    show_progress: bool = True
) -> None:
    """
    Create and save an animated force-velocity plot for a sarcomere model.
    
    Uses blitting and pre-computation for optimal performance (~6-10x faster than naive approach).
    
    Parameters
    ----------
    model : Model
        The sarcomere model object containing data and parameters
    forces : np.ndarray
        Array of forces with shape (N_sarcomeres, N_timepoints)
    output_path : str
        Full path where the animation MP4 will be saved
    tlim : tuple of float, optional
        Time limits for overlay plot (default: (0, 5))
    ylim : tuple of float or None, optional
        Y-axis limits for length plot (default: None)
    vlim : tuple of float, optional
        Velocity limits for force-velocity relation (default: (-9, 5))
    force_ylim : tuple of float, optional
        Y-axis limits for force plot (default: (-1.5, 1.5))
    figsize : tuple of float, optional
        Figure size in inches (default: (2.8, 3.5))
    dpi : int, optional
        Resolution for saved video (default: 300)
    fps : int, optional
        Frames per second for output video (default: 50)
    interval : int, optional
        Delay between frames in milliseconds (default: 20)
    colormap : str, optional
        Matplotlib colormap name (default: 'nipy_spectral')
    preset : str, optional
        FFmpeg encoding preset: 'ultrafast', 'fast', 'medium', 'slow' (default: 'fast')
    bitrate : int, optional
        Video bitrate (default: 5000)
    show_progress : bool, optional
        Whether to show tqdm progress bar (default: True)
    
    Returns
    -------
    None
        Saves animation to output_path
    
    Examples
    --------
    >>> from mypackage import plots
    >>> plots.animate_force_velocity(
    ...     model=model_soft,
    ...     forces=forces_soft,
    ...     output_path='output/animation_soft.mp4',
    ...     dpi=400,
    ...     preset='slow'
    ... )
    Saving animation: 100%|████████| 5000/5000 [01:23<00:00, 60.12frames/s]
    Animation saved to: output/animation_soft.mp4
    """
    # Pre-compute data
    act = model.data['act']
    time_data = model.data['time']
    vel = model.data['vel']
    
    # Handle both dict and dataclass for model_params
    if hasattr(model.model_params, 'N'):
        N = model.model_params.N
    else:
        N = model.model_params['N']
    
    cm_colors = plt.get_cmap(colormap)(np.linspace(0, 1, N))
    
    # Pre-compute force-velocity curves for all time points (major speedup)
    v = np.linspace(vlim[0], vlim[1], 100)
    fvr_all = np.array([
        model.force_vel_rel_sarc(-v, x=0, h=1, act=act[t]) 
        for t in range(len(act))
    ])
    
    # Create figure
    mosaic = """
    A
    B
    B
    """
    fig, axs = plt.subplot_mosaic(
        mosaic=mosaic, 
        figsize=figsize, 
        constrained_layout=True
    )
    axs['A'].set_prop_cycle('color', list(cm_colors))
    
    # Plot static elements - Panel A (length trajectories)
    plot_overlay_length(axs['A'], model, tlim=tlim, y_lim=ylim)
    
    # Plot static elements - Panel B (force-velocity)
    axs['B'].plot(vel.T, -forces.T, lw=0.1, c='r', zorder=-1, alpha=0.5)
    axs['B'].plot((-1000, -1000), (-1000, -2000), lw=0.5, c='r', label='Individual')
    axs['B'].plot(
        np.mean(vel, axis=0), 
        -np.mean(forces, axis=0), 
        c='k', lw=1, label='Average'
    )
    axs['B'].axhline(0, lw=1, c='k', linestyle='--')
    axs['B'].axvline(0, lw=1, c='k', linestyle='--')
    
    # Configure Panel B
    axs['B'].set_xlim(vlim)
    axs['B'].set_ylim(force_ylim)
    polish_xticks(axs['B'], 2, 1)
    polish_yticks(axs['B'], 0.5, 0.25)
    axs['B'].set_xlabel('Velocity')
    axs['B'].set_ylabel('Force')
    axs['B'].legend(fontsize='small')
    
    # Create animated artists (will be updated each frame)
    line = axs['A'].axvline(0, lw=1, c='k', linestyle='-', animated=True)
    fvr_plot, = axs['B'].plot(
        v, fvr_all[0], c='royalblue', 
        label='F-V relation', animated=True
    )
    scatter = axs['B'].scatter(
        vel[:, 0], -forces[:, 0], 
        s=10, c=cm_colors, zorder=3, animated=True
    )
    
    # Update function (called for each frame)
    def update(frame):
        """Update animated elements for current frame."""
        line.set_xdata([time_data[frame], time_data[frame]])
        fvr_plot.set_ydata(fvr_all[frame])
        scatter.set_offsets(np.c_[vel[:, frame], -forces[:, frame]])
        return line, fvr_plot, scatter
    
    # Create animation with blitting enabled (6-10x speedup)
    animation = FuncAnimation(
        fig, 
        update, 
        frames=len(act),
        interval=interval,
        blit=True,
        repeat=False
    )
    
    # Setup optimized FFmpeg writer
    writer = FFMpegWriter(
        fps=fps,
        codec='h264',
        bitrate=bitrate,
        extra_args=['-preset', preset]
    )
    
    # Save with optional progress bar
    if show_progress:
        with tqdm(total=len(act), desc='Saving animation', unit='frames') as pbar:
            animation.save(
                output_path, 
                writer=writer, 
                dpi=dpi,
                progress_callback=lambda i, n: pbar.update(1)
            )
        print(f"Animation saved to: {output_path}")
    else:
        animation.save(output_path, writer=writer, dpi=dpi)
    
    plt.close(fig)

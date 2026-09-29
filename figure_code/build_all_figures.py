"""Read frozen results, export auditable source data and draw Figures 2-4.

No optimizer is imported or executed. Figure 1 is copied, never rendered here.
Only the frozen Figure 2 forward model is deterministically evaluated, using
the original seed 41001, C, J transform, profiles and four frequencies.
"""
from pathlib import Path
import argparse, hashlib, json, shutil, itertools, platform
import sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import style
from style import *

BASE=Path(__file__).resolve().parents[1]
NOISE=[0,.0001,.0003,.001,.003]
NOISE_LABELS=['0','0.01','0.03','0.1','0.3']
MECH_KEYS=['reference','low_edge_ratio','high_edge_ratio','low_stiffness','high_stiffness','low_storage','high_storage','constant_edge_mapping','heterogeneous_edges']
MECH_LABELS=['Reference','Weak coupling','Strong coupling','Stiffness ×0.5','Stiffness ×2','Storage ×0.5','Storage ×1.5','Constant edge weights','More heterogeneous\nedges']
INT_KEYS=['ideal_control','repeated_all_queries_x3','shunt_bias_minus_5pct','shunt_bias_plus_5pct','shunt_bias_minus_15pct','shunt_bias_plus_15pct','root_gain_plus_5pct','finite_clamp_50x_median_diagonal','quantization_0p1pct_baseline','quantization_0p5pct_baseline']
INT_LABELS=['Ideal operations\n77 reads','Three repeats\n231 reads','Shunt bias −5%','Shunt bias +5%','Shunt bias −15%','Shunt bias +15%','Root gain +5%','Finite clamp','Quantization 0.1%','Quantization 0.5%']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding='utf8'))
def quant(v):
    v=pd.Series(v,dtype=float);v=v[np.isfinite(v)]
    return dict(zip(['q25','median','q75','p90'],v.quantile([.25,.5,.75,.9],interpolation='linear').values))
def csv(df,p):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False,encoding='utf-8',na_rep='NA',float_format='%.17g')

class Data:
    def __init__(self,root,out):
        self.root,self.out=Path(root),Path(out);self.rows={k:[] for k in range(1,5)};self.qa=[];self.inputs={}
        self.known_tree_morphology=self.load('results/k8_morphology_sensitivity_replicates.csv')
        self.allprimary=self.load('results/k9_practical_replicates.csv');self.primary=self.allprimary.query('n == 5').copy()
        self.cutoff=self.load('results/k9_support_cutoff_replicates.csv')
        self.mech=self.load('results/k9_mechanics_sensitivity.csv');self.inter=self.load('results/k9_intervention_imperfections.csv')
        self.motif=self.load('results/k8_motif_selection_freeze.json')['selected']
        self.graph=self.load('results/public_subgraph_selection.json')['graphs']['E3']['k9']['5']
        self.exact=self.load('results/k8_exact_checks.json')['checks']
        self.alias_csv=self.load('figures/source/V3_Figure2_k8_alias_frequency.csv')
    def load(self,rel):
        p=self.root/rel;self.inputs[str(p.resolve())]=sha(p)
        return read_json(p) if p.suffix=='.json' else pd.read_csv(p)
    def check(self,name,ok,actual='',expected=''):
        self.qa.append(dict(check=name,status='PASS' if ok else 'FAIL',actual=str(actual),expected=str(expected)))
    def source(self,fig,panel,record_type,rel,**vals):
        p=self.root/rel if (self.root/rel).exists() else BASE/rel
        self.rows[fig].append(dict(figure=fig,panel=panel,record_type=record_type,source_path=rel,source_sha256=sha(p),**vals))
    def population(self,fig,panel,df,rel,groups,metrics,filter_text):
        """Raw values and each plotted summary share explicit units/provenance."""
        for i,r in df.iterrows():
            identity='|'.join(str(r[c]) for c in ['movie','n','design','noise_fraction','setting_id','ensemble','scenario','draw_id','draw','replicate','seed','cutoff'] if c in r)
            for metric in metrics:
                v=r[metric];factor=100 if 'error' in metric else 1
                self.source(fig,panel,'raw',rel,**r.drop(labels=metrics).to_dict(),source_row_index=int(i),case_id=identity,
                    metric=metric,raw_value=v,display_value=v*factor,display_unit='percent' if factor==100 else 'unitless',
                    transform='100 * raw_value' if factor==100 else 'identity',filter=filter_text,
                    nominal_query_count=20 if r.get('design')=='minimal_exact_budget' else r.get('query_count',r.get('query_or_read_count',np.nan)),
                    actual_query_count=r.get('query_count',r.get('query_or_read_count',np.nan)))
        for key,g in df.groupby(groups,dropna=False,sort=False):
            key=key if isinstance(key,tuple) else (key,)
            for metric in metrics:
                q=quant(g[metric]);factor=100 if 'error' in metric else 1
                self.source(fig,panel,'summary',rel,**dict(zip(groups,key)),metric=metric,filter=filter_text,
                    total_count=len(g),finite_count=int(np.isfinite(g[metric]).sum()),converged_count=int(g.fit_success.sum()) if 'fit_success' in g else np.nan,
                    display_unit='percent' if factor==100 else 'unitless',transform='100 * raw_value' if factor==100 else 'identity',
                    **{f'raw_{k}':v for k,v in q.items()},**{f'display_{k}':v*factor for k,v in q.items()})

def forward(d):
    """Literal deterministic forward equations for the known-tree coded-profile example."""
    xy=np.array(d.motif['node_positions_px']);edges=d.motif['edges']
    lengths=np.array([np.linalg.norm(xy[u]-xy[v]) for u,v in edges]);coupling=5*np.median(lengths)/lengths
    local=np.clip(np.random.default_rng(41001).normal(16,1.3,4)/16,.65,1.35)
    B=np.zeros((4,3))
    for k,(u,v) in enumerate(edges): B[u,k],B[v,k]=-1,1
    L=np.diag(local)+B@np.diag(coupling)@B.T; C=np.diag([.92,1.08,.86,1.15])
    ci=np.diag(1/np.sqrt(np.diag(C)));J=-ci@L@ci
    perm=[0,1,3,2];Jswap=J[np.ix_(perm,perm)];P=np.array([[0,1,0,0],[0,0,1,0]])
    freqs=np.array([.35,.75,1.2,1.75]);models=[]
    for name,j in [('Original',J),('Daughter-swapped',Jswap)]:
        h=[];y=[]
        for w in freqs:
            G=np.linalg.inv(1j*w*np.eye(4)-j); h.append(G[0,0])
            background=.4*(1+w/(1+w));y.append((background+P@(np.abs(G[0])**2))-background)
        models.append((np.array(h),np.array(y)))
        for idx,w in enumerate(freqs):
            for metric,v in [('coherent_real',h[idx].real),('coherent_imag',h[idx].imag),('coherent_abs',abs(h[idx])),('added_root_power_P2',y[idx][1])]:
                d.source(2,'b' if metric.startswith('coherent') else 'c','forward','code/run_k8_morphology_validation.py',
                    model=name,frequency=w,metric=metric,raw_value=v,display_value=v,display_unit='dimensionless',
                    transform='identity',method='Frozen deterministic forward; seed 41001; no fitting')
    for name,array in [('J',J),('J_daughter_swapped',Jswap),('L',L),('C',C),('profiles',P)]:
        for (i,j),v in np.ndenumerate(array):d.source(2,'a/c','model','code/run_k8_morphology_validation.py',metric=name,row=i,column=j,raw_value=v)
    for i,(xy0,original) in enumerate(zip(xy,d.motif['node_order'])):
        d.source(2,'a','node','results/k8_motif_selection_freeze.json',node_index=i,node_label=['r','1','2','3'][i],original_node=original,x_px=xy0[0],y_px=xy0[1])
    for k,(u,v) in enumerate(edges):d.source(2,'a','edge','results/k8_motif_selection_freeze.json',node_u=u,node_v=v,persistence=d.motif['detection_persistence'][k],length_px=lengths[k],coupling=coupling[k])
    for r in d.exact:
        for metric in ['coherent_daughter_swap_max_abs','labelled_profile_swap_max_abs','known_tree_fit_relative_L_error','exact_inverse_relative_L_error']:
            d.source(2,'d' if 'error' in metric else 'b/c','exact_record','results/k8_exact_checks.json',mapping=r['mapping'],metric=metric,raw_value=r[metric],display_unit='ratio',status=r['exact_inverse_status'] if metric.startswith('exact_') else 'retained')
    original,swapped=models;obs=d.alias_csv
    for arr,col in [(np.abs(original[0]),'coherent_abs'),(np.abs(swapped[0]),'coherent_swap_abs'),(original[1][:,1],'discriminating_profile_power_difference'),(swapped[1][:,1],'discriminating_profile_power_difference_swap')]:
        delta=np.max(np.abs(arr-obs[col].values));d.check('forward CSV '+col,delta<6e-11,delta,'<6e-11; frozen CSV printed precision')
    delta=np.max(np.abs(original[0]-swapped[0])); d.check('complex alias frozen check',abs(delta-d.exact[0]['coherent_daughter_swap_max_abs'])<2e-16,delta)
    d.check('profile separation frozen check',np.isclose(np.max(np.abs(original[1]-swapped[1])),d.exact[0]['labelled_profile_swap_max_abs'],atol=1e-15))
    d.check('fixed profiles and motif',P.tolist()==[[0,1,0,0],[0,0,1,0]] and d.motif['node_order']==[448,446,408,463])
    return freqs,models

def draw_graph(a,xy,edges,junction=None,labels=True,candidates=False):
    xy=np.asarray(xy);a.set_aspect('equal');a.axis('off')
    for u,v in edges:a.plot(xy[[u,v],0],xy[[u,v],1],color=PALE if candidates else INK,lw=1.2,ls='--' if candidates else '-',zorder=1)
    for i,(x,y) in enumerate(xy):
        col=CORAL if i==0 else GOLD if i==junction else TEAL
        a.plot(x,y,marker='s' if i==0 else 'o',ms=5.5,color=col,zorder=3)
        offset=(-9,-2) if junction is not None and i==3 else (7,-5) if i==junction else (5,3)
        if labels:a.annotate('r' if i==0 else str(i),(x,y),xytext=offset,textcoords='offset points',color=CORAL if i==0 else INK,fontsize=7.2,zorder=4)
    span=max(np.ptp(xy[:,0]),np.ptp(xy[:,1]));a.set_xlim(xy[:,0].min()-.18*span,xy[:,0].max()+.24*span);a.set_ylim(xy[:,1].min()-.18*span,xy[:,1].max()+.2*span)

def fig2(d,freqs,models):
    f=page(180)
    title(f,5,5,'a','A morphology-defined\nbranch')
    title(f,60,5,'b','Identical coherent\nresponses')
    title(f,119,5,'c','A fixed profile separates\nthe aliases')
    ax=axes(f,8,19,43,40);draw_graph(ax,d.motif['node_positions_px'],d.motif['edges'],1)
    text(f,10,59,'E5 · known labelled tree',7.2)
    text(f,10,64,'Daughter exchange: 2 ↔ 3',7.2)
    text(f,10,69,r"$J'=\Pi J\Pi^{\mathsf{T}}$; C held fixed",7.2)
    text(f,10,74,r"$L'=-C^{1/2}J'C^{1/2}$",7.2)
    handles=[Line2D([],[],color=TEAL,marker='o',ls='-',ms=3.5,label='Original'),Line2D([],[],color=GOLD,marker='D',mfc='white',ls='--',ms=5,label='Daughter-swapped')]
    f.legend(handles=handles,loc='upper center',bbox_to_anchor=(.67,1-17/180),ncol=2,frameon=False,columnspacing=1.2,handlelength=1.7,fontsize=7)
    for row,part in enumerate(['real','imag']):
        a=axes(f,69,26+row*23,42,18)
        for i,(h,y) in reversed(list(enumerate(models))):
            a.plot(freqs,getattr(h,part),color=[TEAL,GOLD][i],ls=['-','--'][i],marker=['o','D'][i],ms=[3,5.5][i],mfc=TEAL if i==0 else 'white',mew=1.1,zorder=4-i)
        a.set_ylabel(['Re hᵣ','Im hᵣ'][row]);a.set_xticks(freqs,labels=['0.35','0.75','1.20','1.75']);a.tick_params(labelbottom=row==1)
    text(f,69,73,'ω (dimensionless)',7.2)
    text(f,64,79,r'max $|\Delta h_r|=6.2\times10^{-17}$',7.2)
    a=axes(f,130,33,43,34)
    for i,(h,y) in enumerate(models):a.plot(freqs,y[:,1],color=[TEAL,GOLD][i],ls=['-','--'][i],marker=['o','D'][i],ms=[3.5,4.5][i],mfc=TEAL if i==0 else 'white',mew=1.1)
    a.set_ylabel(r'Added root power, $y_2$');a.set_xticks(freqs,labels=['0.35','0.75','1.20','1.75']);a.set_ylim(0,.044)
    a.set_yticks([0,.02,.04]);a.set_xlabel('ω (dimensionless)',labelpad=4)
    mask=axes(f,130,22,41,8);mask.axis('off');mask.set_xlim(-1.2,4);mask.set_ylim(-.7,1.6)
    mask.text(-1.15,.3,r'$P_2$',va='center')
    for i in range(4):
        mask.add_patch(Rectangle((i,0),.8,.65,facecolor=GOLD if i==2 else 'white',edgecolor=INK,lw=1));mask.text(i+.4,1.05,['r','1','2','3'][i],ha='center',color=CORAL if i==0 else INK,fontsize=7)
    text(f,125,79,r'max $|\Delta y_2|=0.00726$',7.2)
    text(f,119,84,'One of two fixed profiles shown',7)
    line(f,5,91,175,91)
    title(f,5,95,'d','Information and numerical reconstruction')
    text(f,8,103,'Routine',7.2,fontweight='bold');text(f,94,103,'Inverse-length edges',7.2,ha='center',fontweight='bold');text(f,149,103,'Constant edges',7.2,ha='center',fontweight='bold')
    text(f,8,110,'Known-tree fit · relative L error',7.2);text(f,94,110,r'$2.95\times10^{-14}$',7.2,ha='center');text(f,149,110,r'$4.78\times10^{-14}$',7.2,ha='center')
    text(f,8,116,'Exact finite-frequency routine',7.2);text(f,94,116,'Precision refusal',7.2,ha='center');text(f,149,116,'Precision refusal',7.2,ha='center')
    line(f,5,123,175,123)
    title(f,5,126,'e','Sensitivity to profile imperfections')
    families=[('Profile dose',['0.25×','0.50×','1.00×'],range(1,4)),('Calibration bias',['−15','−5','+5','+15'],range(4,8)),('Spatial leakage',['5','15','30'],range(8,11)),('Repeated reads',['1','3','10'],range(11,14))]
    # This panel is 38 mm high; notes and x units are outside the data area.
    for k,(name,labels,ids) in enumerate(families):
        x=17+k*42; a=axes(f,x,139,30,29)
        text(f,x,133,name,7.2,fontweight='bold')
        for j,setting in enumerate(ids):
            g=d.known_tree_morphology[d.known_tree_morphology.setting_id==f'S{setting:02d}'].sort_values('seed');v=g.relative_L_error.values*100
            jitter=np.random.default_rng(7926).uniform(-.2,.2,len(g))
            a.scatter(j+jitter,v,s=7,facecolors='none',edgecolors=TEAL,lw=1,alpha=.55,zorder=2)
            q=quant(v);a.errorbar(j,q['median'],yerr=[[q['median']-q['q25']],[q['q75']-q['median']]],fmt='o',ms=3,color=INK,capsize=2,lw=1.2,zorder=4)
            fail=~g.fit_success.values
            if fail.any():a.scatter(j+jitter[fail],v[fail],marker='x',s=20,color=CORAL,lw=1.2,zorder=5)
        a.set_ylim(0,100);a.set_xlim(-.55,len(labels)-.45);a.set_xticks(range(len(labels)),labels);a.set_yticks([0,50,100]);a.tick_params(labelleft=k==0)
        if k==0:a.set_ylabel('Relative L error (%)')
        text(f,x,176,['0.1% noise','Bias (%) · 0.1% noise','Leakage (%) · 0.1% noise','0.3% noise'][k],6.8)
    # Count is explicit, while the finite nonconverged observation remains plotted.
    text(f,174,126,'20 seeds/setting; × nonconverged (19/20 at one repeat)',6.8,ha='right')
    save(f,d.out,2)

def summary_curve(a,df,metric,group_values,group_col,kind='iqr',percent=False):
    for j,design in enumerate(DESIGNS):
        qs=[quant(df[(df.design==design)&np.isclose(df[group_col],x)][metric]*(100 if percent else 1)) for x in group_values]
        mid=np.array([q['median'] for q in qs]);lo=np.array([q['q25'] for q in qs]);hi=np.array([q['q75' if kind=='iqr' else 'p90'] for q in qs])
        x=np.arange(len(group_values))
        a.errorbar(x,mid,yerr=[mid-lo if kind=='iqr' else np.zeros_like(mid),hi-mid],
            color=COLORS[design],marker=MARKERS[design],mfc='white' if j==0 else TEAL,ms=4,
            ls='--' if j==0 else '-',lw=1.4,capsize=2.5,zorder=3+j)

def fig3(d):
    f=page(182)
    title(f,5,5,'a','Recovery without supplied adjacency')
    # A schematic layout of the fixed five states, never the hidden adjacency.
    xy=np.array([[-1,0],[-.25,.8],[.85,.5],[.45,-.85],[-.45,-.7]])
    a=axes(f,7,15,24,21);draw_graph(a,xy,list(itertools.combinations(range(5),2)),candidates=True)
    text(f,7,38,'Known nodes + root',7.1)
    labels=[(40,'Simulated intervention\nrecords'),(89,'All-candidate-edge\nfit'),(143,'L, C and\nedge support')]
    for x,s in labels:text(f,x,22,s,7.4)
    for x0,x1 in [(31,38),(78,86),(132,141)]:
        f.add_artist(matplotlib.patches.FancyArrowPatch((x0/180,1-26/182),(x1/180,1-26/182),transform=f.transFigure,arrowstyle='->',mutation_scale=8,lw=1,color=INK))
    text(f,42,38,'Morphology-derived adjacency: simulation and held-out scoring only',7, color=INK)
    text(f,7,45,'Ideal exact route: shunts → L; retained-state clamps → C',7.2)
    text(f,7,51,'Practical fit: unknown adjacency and C; positive grounded-Laplacian model with finite bounds',7)
    line(f,5,58,175,58)
    title(f,5,62,'b','Three within-contract designs')
    for y,label,reads,detail,col in [(73,'Exact construction','20 nominal','Ideal adaptive schedule',INK),(84,'Moderate design','37 reads','Narrower shunt coverage',GOLD),(95,'Richer design','77 reads','Broader shunt coverage',TEAL)]:
        text(f,8,y,label,7.3,fontweight='bold');text(f,77,y,reads,7.3,ha='right',fontweight='bold');text(f,8,y+4.5,detail,7)
    text(f,8,108,'Read count and intervention\ncoverage both change.',7.1)
    title(f,85,62,'c','Mechanical recovery at 0.1% noise')
    sub=d.primary[np.isclose(d.primary.noise_fraction,.001)&d.primary.design.isin(DESIGNS)]
    for k,metric in enumerate(['L_relative_error','C_relative_error']):
        a=axes(f,97+k*43,77,29,33)
        for j,design in enumerate(DESIGNS):
            g=sub[sub.design==design];v=g[metric].values*100;q=quant(v);offset=np.random.default_rng(8832).uniform(-.17,.17,len(g))
            a.scatter(j+offset,v,s=10,marker='o' if k==0 else 's',facecolors=COLORS[design] if k==0 else 'white',edgecolors=COLORS[design],lw=1,alpha=.7)
            a.errorbar(j,q['median'],yerr=[[q['median']-q['q25']],[q['q75']-q['median']]],fmt='o' if k==0 else 's',mfc=INK if k==0 else 'white',color=INK,ms=4,capsize=3,zorder=5)
        a.set_yscale('log');a.set_ylim(.05,500);a.set_yticks([.1,1,10,100],labels=['0.1','1','10','100']);a.yaxis.set_minor_locator(NullLocator());a.set_xlim(-.5,1.5);a.set_xticks([0,1],['37','77']);a.set_xlabel('Reads')
        a.set_title(['Stiffness, L','Storage, C'][k],fontsize=7.5,pad=5)
        if k==0:a.set_ylabel('Relative error (%)')
    text(f,91,120,'24 cases/design · median and IQR',7)
    line(f,5,123,175,123)
    title(f,5,127,'d','Support recovery under noise')
    title(f,96,127,'e','Fixed-cutoff sensitivity')
    a=axes(f,17,140,64,24);summary_curve(a,d.primary,'topology_F1',NOISE,'noise_fraction')
    a.plot(0,1,'s',ms=6,mfc='none',mec=INK,mew=1.1,zorder=7)
    a.set_ylim(0,1.08);a.set_yticks([0,.5,1]);a.set_ylabel('Support F1');a.set_xticks(range(5),NOISE_LABELS);a.set_xlabel('Threshold noise (%)',labelpad=3)
    text(f,17,135,'Cutoff = 0.25',7)
    # Status strip has its own explicit row outside the F1 axes, never data points.
    text(f,5,175,'Exact:',6.8)
    for i,label in enumerate(['Recovered','NA','NA','NA','NA']):text(f,20+61*(i/4),175,label,6.8,ha='center')
    a=axes(f,110,140,63,24);sub=d.cutoff[(d.cutoff.n==5)&np.isclose(d.cutoff.noise_fraction,.001)]
    summary_curve(a,sub,'topology_F1',[.15,.2,.25,.3,.35],'cutoff')
    a.axvline(2,color=INK,lw=1,ls=':',zorder=0);a.set_ylim(0,1.08);a.set_yticks([0,.5,1]);a.set_ylabel('Support F1');a.set_xticks(range(5),['0.15','0.20','0.25','0.30','0.35']);a.set_xlabel('Edge-weight cutoff',labelpad=3)
    text(f,110,135,'0.1% noise · primary cutoff 0.25',7)
    f.legend(handles=design_handles(),loc='upper center',bbox_to_anchor=(.67,1-129/182),ncol=2,frameon=False,fontsize=7,handlelength=1.8)
    save(f,d.out,3)

def forest(f,d,x,y,w,h,df,key,keys,labels):
    a=axes(f,x,y,w,h);a.set_xscale('log');a.set_xlim(.7,400);a.set_ylim(len(keys)-.4,-.6)
    a.set_xticks([1,10,100],labels=['1','10','100']);a.xaxis.set_minor_locator(NullLocator());a.set_yticks([])
    a.spines['left'].set_visible(False);a.set_xlabel('Relative error (%)',labelpad=5)
    for i,(k,label) in enumerate(zip(keys,labels)):
        for j,metric in enumerate(['L_relative_error','C_relative_error']):
            q=quant(df[df[key]==k][metric]*100);yy=i+[-.15,.15][j]
            a.plot([q['median'],q['p90']],[yy,yy],color=TEAL,ls=['-','--'][j],lw=1.4)
            a.plot(q['p90'],yy,marker='|',ms=5,color=TEAL,mew=1)
            a.plot(q['median'],yy,marker=['o','s'][j],mfc=TEAL if j==0 else 'white',mec=TEAL,ms=4,mew=1.1)
        # Row labels placed in the left gutter in data-aligned coordinates.
        a.text(-.07,i,label,transform=a.get_yaxis_transform(),ha='right',va='center',fontsize=7,
               fontweight='bold' if k in ['reference','low_edge_ratio'] else 'normal',clip_on=False,linespacing=1.05)
    return a

def fig4(d):
    f=page(190)
    title(f,5,5,'a','Stiffness recovery')
    title(f,95,5,'b','Storage recovery')
    f.legend(handles=design_handles(),loc='upper center',bbox_to_anchor=(.5,.93),ncol=2,frameon=False,fontsize=7.2,handlelength=2)
    for k,metric in enumerate(['L_relative_error','C_relative_error']):
        a=axes(f,18+k*88,23,65,48);summary_curve(a,d.primary,'L_relative_error' if k==0 else 'C_relative_error',NOISE,'noise_fraction',kind='p90',percent=True)
        a.set_yscale('log');a.set_ylim(1e-4,500);a.set_yticks([1e-4,.01,1,100]);a.yaxis.set_minor_locator(NullLocator())
        a.set_ylabel('Relative '+['L','C'][k]+' error (%)');a.set_xticks(range(5),NOISE_LABELS);a.set_xlabel('Threshold noise (%)')
    text(f,18,81,'24 cases/design/noise level · markers: median; upper endpoints: 90th percentile',7.1)
    line(f,5,89,175,89)
    title(f,5,94,'c','Mechanical conditions\nshape recovery')
    title(f,96,94,'d','Intervention imperfections')
    text(f,10,107,'77 reads · 0.1% noise · 36 cases/condition',7)
    text(f,100,107,'0.1% noise · 5 replicates/condition',7)
    f.legend(handles=metric_handles(),loc='upper center',bbox_to_anchor=(.5,1-112/190),ncol=2,frameon=False,fontsize=7.3,handlelength=2)
    forest(f,d,47,123,39,51,d.mech,'ensemble',MECH_KEYS,MECH_LABELS)
    forest(f,d,139,123,35,51,d.inter,'scenario',INT_KEYS,INT_LABELS)
    text(f,7,187,'Intervals: median → 90th percentile; empirical sensitivity, not confidence intervals',7)
    save(f,d.out,4)

def schedules(d):
    rel='code/run_practical_k9_validation.py';rows=[]
    for design,singles,pairs in [('moderate_redundancy',[.5,2],[1]),('high_redundancy',[.5,2,10,50],[.5,2,10])]:
        rec=[]
        for growth in [0,1]:
            rec.extend([dict(growth=growth,shunt='[0,0,0,0,0]',retained='[0,1,2,3,4]',kind='baseline') for _ in range(2)])
            for nodes,loads in [(list(itertools.combinations(range(1,5),1)),singles),(list(itertools.combinations(range(1,5),2)),pairs)]:
                for ns in nodes:
                    for load in loads:
                        q=[0]*5
                        for node in ns:q[node]=load
                        rec.append(dict(growth=growth,shunt=json.dumps(q),retained='[0,1,2,3,4]',kind='shunt'))
        for kept in [[0]]+[[0,i] for i in range(1,5)]:rec.append(dict(growth=1,shunt='[0,0,0,0,0]',retained=json.dumps(kept),kind='clamp'))
        d.check('schedule '+design,len(rec)==({'moderate_redundancy':37,'high_redundancy':77}[design]),len(rec))
        for i,r in enumerate(rec):d.source(3,'b','query_schedule',rel,design=design,query_index=i,nominal_query_count=len(rec),**r)
    d.source(3,'b','design',rel,design='minimal_exact_budget',nominal_query_count=20,method='Ideal adaptive n(n+3)/2 contract; actual acquired reads per retained case, refusal can stop early')
    for i,(original,pos) in enumerate(zip(d.graph['node_order'],d.graph['node_positions_px'])):d.source(3,'a','node','results/public_subgraph_selection.json',node_index=i,original_node=original,x_px=pos[0],y_px=pos[1],role='known identity; display positions schematic')
    for u,v in itertools.combinations(range(5),2):d.source(3,'a','candidate_edge','results/public_subgraph_selection.json',node_u=u,node_v=v,role='all unordered pairs; no supplied adjacency')
    for u,v in d.graph['edges']:d.source(3,'a','held_out_truth_edge','results/public_subgraph_selection.json',node_u=u,node_v=v,role='simulation and scoring only; not drawn as fit input')

def export_populations(d):
    d.population(2,'e',d.known_tree_morphology,'results/k8_morphology_sensitivity_replicates.csv',['setting_id'],['relative_L_error'],'all 260 fits; finite nonconverged retained')
    d.population(3,'c',d.primary[np.isclose(d.primary.noise_fraction,.001)&d.primary.design.isin(DESIGNS)],'results/k9_practical_replicates.csv',['design'],['L_relative_error','C_relative_error'],'n==5; noise_fraction==0.001; practical designs')
    d.population(3,'d',d.primary,'results/k9_practical_replicates.csv',['design','noise_fraction'],['topology_F1'],'n==5; primary support cutoff 0.25')
    d.population(3,'e',d.cutoff[(d.cutoff.n==5)&np.isclose(d.cutoff.noise_fraction,.001)&d.cutoff.design.isin(DESIGNS)],'results/k9_support_cutoff_replicates.csv',['design','cutoff'],['topology_F1'],'n==5; noise_fraction==0.001; practical; fixed replay classifications')
    for p,m in [('a','L_relative_error'),('b','C_relative_error')]:d.population(4,p,d.primary[d.primary.design.isin(DESIGNS)],'results/k9_practical_replicates.csv',['design','noise_fraction'],[m],'n==5; practical designs; retained coefficients')
    d.population(4,'c',d.mech,'results/k9_mechanics_sensitivity.csv',['ensemble'],['L_relative_error','C_relative_error'],'all 324; noise_fraction==0.001; 77 reads')
    d.population(4,'d',d.inter,'results/k9_intervention_imperfections.csv',['scenario'],['L_relative_error','C_relative_error'],'all 50; E3 five-node graph; noise_fraction==0.001')
    for n in ['CROP_NODES.csv','CROP_EDGES.csv']:
        rel='figure1_reproduction/source/records/'+n
        for _,r in pd.read_csv(BASE/rel).iterrows():d.source(1,'a','node' if 'NODES' in n else 'edge',rel,**r.to_dict())
    rel='figure1_reproduction/source/inputs/schematic_layout.json';layout=read_json(BASE/rel)
    for name in ['path','branch','unknown']:
        for i,xy in enumerate(layout[name]['positions']):d.source(1,'b','schematic_node',rel,regime=name,node_index=i,x_schematic=xy[0],y_schematic=xy[1],provenance=layout[name]['provenance'])
        edges=layout[name].get('edges',d.motif['edges'] if name=='branch' else d.graph['edges'])
        for u,v in edges:d.source(1,'b','schematic_edge',rel,regime=name,node_u=u,node_v=v,provenance='Frozen topology; schematic coordinates; not measured mechanics')
    for u,v in itertools.combinations(range(5),2):d.source(1,'b','candidate_edge',rel,regime='unknown',node_u=u,node_v=v)
    for fig,rows in d.rows.items():csv(pd.DataFrame(rows),d.out/'source_data'/f'Figure{fig}_source_data.csv')
    # Compact summary views accompany, but never replace, the case-level files.
    for fig,rows in d.rows.items():
        frame=pd.DataFrame(rows)
        if 'summary' in set(frame.record_type):csv(frame[frame.record_type=='summary'].dropna(axis=1,how='all'),d.out/'source_data/raw_and_summaries'/f'Figure{fig}_summaries.csv')

def numerical_checks(d):
    p=d.primary
    d.check('01 primary population',len(p)==360 and p.groupby(['design','noise_fraction']).size().eq(24).all(),len(p),'360; 15 groups of 24')
    d.check('02 3x2x4 primary composition',all((g.movie.nunique(),g.draw_id.nunique(),g.replicate.nunique())==(3,2,4) and not g.duplicated(['movie','draw_id','replicate']).any() for _,g in p.groupby(['design','noise_fraction'])))
    k=d.known_tree_morphology;d.check('03 known-tree morphology population and paired seeds',len(k)==260 and k.groupby('setting_id').size().eq(20).all() and all(set(g.seed)==set(range(91001,91021)) for _,g in k.groupby('setting_id')))
    d.check('03 known-tree noise strata',k[k.condition=='repeated_measurement_noise'].noise_fraction.eq(.003).all() and k[k.condition!='repeated_measurement_noise'].noise_fraction.eq(.001).all())
    failed=k[~k.fit_success];d.check('04 retained finite nonconverged seed',len(failed)==1 and int(failed.iloc[0].seed)==91019 and np.isfinite(failed.relative_L_error).all(),len(failed))
    exact=p[p.design=='minimal_exact_budget'];ref=exact[exact.noise_fraction>0]
    d.check('07 exact refusal NA',len(ref)==96 and ref[['L_relative_error','C_relative_error','topology_F1','topology_precision','topology_recall']].isna().all().all() and (~ref.fit_success).all())
    d.check('08 nominal vs acquired',exact[exact.noise_fraction==0].query_count.eq(20).all() and (ref.query_count<20).any(),sorted(ref.query_count.unique()))
    for design in DESIGNS:
        for metric in ['L_relative_error','C_relative_error']:
            a=[r for r in d.rows[3] if r['record_type']=='summary' and r['panel']=='c' and r['design']==design and r['metric']==metric][0]
            b=[r for r in d.rows[4] if r['record_type']=='summary' and r['panel'] in ['a','b'] and r['design']==design and r['noise_fraction']==.001 and r['metric']==metric][0]
            d.check('09 shared 0.1% '+design+metric,a['raw_median']==b['raw_median'])
    d.check('10 fixed grid and lower quartile',set(d.cutoff.cutoff)=={.15,.2,.25,.3,.35} and quant(d.cutoff[(d.cutoff.design=='high_redundancy')&np.isclose(d.cutoff.noise_fraction,.001)&np.isclose(d.cutoff.cutoff,.15)].topology_F1)['q25']<1)
    d.check('11 retained/replay separation',all('k9_practical_replicates' in r['source_path'] for r in d.rows[4] if r['panel'] in ['a','b']))
    d.check('12 mechanics 9x36',len(d.mech)==324 and d.mech.groupby('ensemble').size().eq(36).all() and d.mech.fit_success.all())
    d.check('12 interventions 10x5',len(d.inter)==50 and d.inter.groupby('scenario').size().eq(5).all() and d.inter.fit_success.all())
    d.check('12 231 reads only complete triplicate',d.inter[d.inter.scenario=='repeated_all_queries_x3'].query_count.eq(231).all() and d.inter[d.inter.scenario!='repeated_all_queries_x3'].query_count.eq(77).all())
    d.check('13 p90 tails retained',quant(d.mech[d.mech.ensemble=='low_edge_ratio'].C_relative_error)['p90']*100>280 and quant(p[(p.design=='moderate_redundancy')&np.isclose(p.noise_fraction,.003)].C_relative_error)['p90']*100>290)
    d.check('14 exact percent conversion',all(np.isclose(r['display_value'],r['raw_value']*100) for f in [2,3,4] for r in d.rows[f] if r['record_type']=='raw' and r['display_unit']=='percent' and np.isfinite(r['raw_value'])))
    d.check('15 populations distinct',len(set(round(v,5) for v in [p[(p.design=='high_redundancy')&np.isclose(p.noise_fraction,.001)].L_relative_error.median(),d.mech[d.mech.ensemble=='reference'].L_relative_error.median(),d.inter[d.inter.scenario=='ideal_control'].L_relative_error.median()]))==3)
    # Compare all 250 independent supplied checkpoints to new raw-data aggregates.
    checkpoints=pd.read_csv(BASE/'inputs/NUMERICAL_CHECKPOINTS.csv')
    for idx,r in checkpoints.iterrows():
        src=r.source;metric=r.metric
        if src.endswith('k8_exact_checks.json'):
            record=next(x for x in d.exact if x['mapping']==r.condition)
            actual='Precision refusal' if r.statistic=='status' and record['exact_inverse_status'].startswith('PrecisionFailure:') else record.get(metric)
        else:
            if 'k8_morphology' in src:g=k[k.setting_id==r.condition.split(':')[0]]
            elif 'k9_practical' in src:g=p[(p.design==r.design)&np.isclose(p.noise_fraction,r.noise_fraction)]
            elif 'cutoff' in src:g=d.cutoff[(d.cutoff.n==5)&(d.cutoff.design==r.design)&np.isclose(d.cutoff.noise_fraction,r.noise_fraction)&np.isclose(d.cutoff.cutoff,float(r.condition.split('=')[1]))]
            elif 'mechanics' in src:g=d.mech[d.mech.ensemble==r.condition]
            elif 'intervention' in src:g=d.inter[d.inter.scenario==r.condition]
            else:raise ValueError(src)
            actual=g[metric].sum() if r.statistic=='count' else quant(g[metric])[r.statistic]
            if r.unit=='percent':actual*=100
        if r.statistic=='status':ok=actual==r.expected_value
        else:ok=np.isclose(float(actual),float(r.expected_value),atol=5e-9,rtol=2e-8,equal_nan=True)
        d.check(f'checkpoint {idx+1:03d}: F{r.figure}{r.panel} {r.condition} {r.design} {metric} {r.statistic}',ok,actual,r.expected_value)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-root',type=Path,default=BASE/'inputs');ap.add_argument('--output-root',type=Path,default=BASE);ap.add_argument('--font-family',choices=['auto','public'],default='auto');args=ap.parse_args()
    style.setup(args.font_family);args.output_root.mkdir(exist_ok=True,parents=True)
    for sub in ['figures','qa','source_data']: (args.output_root/sub).mkdir(exist_ok=True)
    d=Data(args.input_root,args.output_root)
    # Record the frozen numerical inputs and approved Figure 1 assets.
    for p in sorted((BASE/'inputs').rglob('*')):
        if p.is_file():d.inputs[str(p.resolve())]=sha(p)
    for p in sorted((BASE/'figure1_reproduction').rglob('*')):
        if p.is_file():d.inputs[str(p.resolve())]=sha(p)
    freqs,models=forward(d);schedules(d);export_populations(d);numerical_checks(d)
    for ext in ['png','pdf','svg']:
        dst=args.output_root/'figures'/f'Figure1_final.{ext}';src=BASE/'figure1_reproduction'/f'Figure1_final.{ext}'
        if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    fig2(d,freqs,models);fig3(d);fig4(d)
    immutable=[dict(path=p,before_sha256=h,after_sha256=sha(p),status='PASS' if h==sha(p) else 'FAIL') for p,h in d.inputs.items()]
    d.check('16 all read inputs unchanged',all(r['status']=='PASS' for r in immutable))
    csv(pd.DataFrame(immutable),args.output_root/'qa/BUILD_INPUT_IMMUTABILITY_CHECK.csv')
    csv(pd.DataFrame(d.qa),args.output_root/'qa/NUMERICAL_VALIDATION.csv')
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,font=style.ACTIVE_FONT,font_mode=args.font_family,width_mm=180,figure_heights_mm=[180,182,190],dpi=600)
    (args.output_root/'qa/BUILD_ENVIRONMENT.json').write_text(json.dumps(versions,indent=2),encoding='utf8')
    failures=[r for r in d.qa if r['status']!='PASS'];print(f'Built Figures 2-4; {len(d.qa)-len(failures)}/{len(d.qa)} numerical checks passed.')
    if failures:print(json.dumps(failures,indent=2));raise SystemExit(1)
if __name__=='__main__':main()

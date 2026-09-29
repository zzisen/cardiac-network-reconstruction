"""Cross-check the retained tables and already frozen support replay. No fitting."""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
BASE=Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input-root',type=Path,default=BASE/'inputs');ap.add_argument('--output-root',type=Path,default=BASE);args=ap.parse_args()
    r=args.input_root/'results';checks=[]
    def check(name,ok,actual):checks.append(dict(check=name,status='PASS' if ok else 'FAIL',actual=str(actual)))
    primary=pd.read_csv(r/'k9_practical_replicates.csv').query('n==5')
    summary=pd.read_csv(args.input_root/'figures/source/V3_Figure4_practical_noise_summary.csv');errors=[]
    for row in summary.itertuples():
        g=primary[(primary.design==row.design)&np.isclose(primary.noise_fraction,row.noise_fraction)][row.metric]
        errors.extend([abs(g.median()-row.median),abs(g.quantile(.9)-row.p90)])
    check('Frozen primary summary agrees after n=5 filtering',max(errors)<1e-8,float(max(errors)))
    cut=pd.read_csv(r/'k9_support_cutoff_replicates.csv');old=pd.read_csv(r/'k9_support_cutoff_summary.csv');errors=[]
    for row in old.itertuples():
        g=cut[(cut.design==row.design)&np.isclose(cut.noise_fraction,row.noise_fraction)&np.isclose(cut.cutoff,row.cutoff)]
        for metric in ['topology_F1','topology_precision','topology_recall']:
            for stat,q in [('median',.5),('q25',.25),('q75',.75)]:
                a=g[metric].quantile(q);b=getattr(row,metric+'_'+stat)
                if pd.notna(a) or pd.notna(b):errors.append(abs(a-b))
    check('All existing fixed-cutoff summaries agree with raw rows',max(errors)<1e-9,float(max(errors)))
    keys=['movie','n','draw_id','replicate','noise_fraction','design']
    joined=primary.merge(cut[np.isclose(cut.cutoff,.25)],on=keys,suffixes=('_retained','_replay'))
    delta=np.abs(joined.topology_F1_retained-joined.topology_F1_replay).max()
    check('Primary retained vs frozen replay F1',len(joined)==360 and delta<5e-10,dict(rows=len(joined),max_abs=float(delta)))
    e=pd.read_csv(r/'k9_support_cutoff_edge_predictions.csv');errors=[];num=0
    for key,g in e.groupby(keys):
        target=cut.copy()
        for k,v in zip(keys,key):target=target[np.isclose(target[k],v) if k=='noise_fraction' else target[k]==v]
        for row in target.itertuples():
            pred=g.fitted_weight.values>=row.cutoff;truth=g.true_edge.values
            tp=np.sum(pred&truth);fp=np.sum(pred&~truth);fn=np.sum(~pred&truth)
            f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
            errors.append(abs(f1-row.topology_F1));num+=1
    check('Existing edge weights reproduce all scored cutoff classifications',num==1320 and max(errors)<1e-9,dict(scored=num,max_abs=float(max(errors))))
    ref=cut[~cut.prediction_available]
    check('All 480 refused cutoff slots remain NA',len(ref)==480 and ref[['topology_F1','topology_precision','topology_recall']].isna().all().all(),len(ref))
    (args.output_root/'qa').mkdir(exist_ok=True,parents=True)
    pd.DataFrame(checks).to_csv(args.output_root/'qa/INDEPENDENT_AGGREGATE_CHECKS.csv',index=False)
    assert all(x['status']=='PASS' for x in checks),checks
    print('Five independent retained-data cross-checks passed; no optimizer executed.')
if __name__=='__main__':main()

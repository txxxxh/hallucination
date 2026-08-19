#!/usr/bin/env python3
"""Confirmatory summaries for the full single-keyword binding run."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, wilcoxon

ROOT=Path(__file__).resolve().parent/'runs/209_strict_attribute_binding_full'

def bh(p):
 p=np.asarray(p,float);order=np.argsort(p);q=np.empty(len(p));last=1.
 for rank,i in reversed(list(enumerate(order,1))):last=min(last,p[i]*len(p)/rank);q[i]=last
 return q
def stats(rows,rng,boot=10000):
 d=np.asarray([x['binding_effect']for x in rows]);u=np.asarray([x['perturb_u']for x in rows]);means=np.asarray([rng.choice(d,len(d),replace=True).mean()for _ in range(boot)])
 w=wilcoxon(d,alternative='greater');s=spearmanr(d,u)
 return {'n':len(rows),'mean':float(d.mean()),'mean_ci95':[float(np.quantile(means,.025)),float(np.quantile(means,.975))],'median':float(np.median(d)),'fraction_positive':float(np.mean(d>0)),'wilcoxon_greater_p':float(w.pvalue),'binding_vs_perturb_rho':float(s.statistic),'binding_vs_perturb_p':float(s.pvalue),'mean_perturb_u':float(u.mean()),'median_positive_rank':float(np.median([x['rank']for x in rows]))}
def main():
 raw=[json.loads(x)for x in(ROOT/'binding_items.jsonl').open()];rows=[]
 for x in raw:
  a=json.load((ROOT/f"{x['key']}.json").open());rows.append({**x,'rank':a['entity']['rank_positive']})
 rng=np.random.default_rng(42);fields=sorted(set(x['field']for x in rows));report={'all':stats(rows,rng),'top10':stats([x for x in rows if x['rank']<=10],rng),'by_field':{f:stats([x for x in rows if x['field']==f],rng)for f in fields}}
 q=bh([report['by_field'][f]['wilcoxon_greater_p']for f in fields])
 for f,v in zip(fields,q):report['by_field'][f]['bh_q']=float(v)
 cond=[q for x in rows for q in x['conditions']];report['override_task']={'n_conditions':len(cond),'accuracy':float(np.mean([x['correct_margin']>0 for x in cond]))}
 report['interpretation']='No overall single-keyword binding effect. Education and position_held are exploratory positive subgroups but do not survive BH correction.'
 (ROOT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

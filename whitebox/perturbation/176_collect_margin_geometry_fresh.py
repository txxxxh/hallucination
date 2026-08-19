#!/usr/bin/env python3
"""Collect run-173 features on 100 rows disjoint from its exploratory subset."""
import importlib
from pathlib import Path
R=importlib.import_module("174_run_margin_geometry_fixed");B=R.B
old=B.BASE.select_balanced;all_rows,*_=B.BASE.load_rows();used={x["key"]for x in old(all_rows,100,B.BASE.SEED)}
def fresh(rows,n,seed):return old([x for x in rows if x["key"]not in used],n,seed)
B.BASE.select_balanced=fresh;B.OUT=Path(__file__).resolve().parent/"runs/176_known_unknown_margin_geometry_fresh100"
if __name__=="__main__":B.main()

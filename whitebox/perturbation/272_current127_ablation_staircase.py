#!/usr/bin/env python3
"""Exact three-tier current127 ablation under identical grouped OOF folds."""
from __future__ import annotations
import argparse,importlib,json
from pathlib import Path
import numpy as np

ablation=importlib.import_module("161_current127_feature_ablation")
RUNS=Path(__file__).resolve().parent/"runs"
# ch(pred): baseline, 5 u_s, 5 normalized u_s, then max u_s at column 11.
ablation.CURVES["max_us_1d"]=np.asarray([11])

def configs():
    return [
        {"name":"max_s_u_s","curve":"max_us_1d","hidden":[],"hdim":0,"ldim":0},
        {"name":"span_distribution_47","curve":"full47","hidden":[],"hdim":0,"ldim":0},
        {"name":"representation_pca_only_80","curve":"none0","hidden":[0,1,2,3],"hdim":8,"ldim":48},
        {"name":"span_distribution_plus_pca_127","curve":"full47","hidden":[0,1,2,3],"hdim":8,"ldim":48},
    ]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cache",type=Path,default=RUNS/"159_scientist_classgrad_sentence_current127")
    p.add_argument("--output",type=Path,default=RUNS/"272_current127_ablation_staircase.json")
    a=p.parse_args();ablation.configs=configs
    report=ablation.run(a.cache)
    report["protocol_note"]="Same candidate-grouped 3x5 OOF splits, LR, fold-local scaling, and fold-local PCA for all tiers."
    report["dimension_identity"]="1 -> 47 -> 47 + 4*8 + 48 = 127"
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":main()

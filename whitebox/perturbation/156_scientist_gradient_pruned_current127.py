#!/usr/bin/env python3
"""Scientist current127 with pure first-order gate-gradient coarse screening."""
import argparse, importlib, json
import numpy as np

m = importlib.import_module('152_scientist_attention_pruned_current127')
m.CACHE = m.RUNS / '156_scientist_gradient_pruned_current127'
m.OUT = m.RUNS / '156_scientist_gradient_pruned_current127_report.json'

def gradient_shortlist(att, prep, spans, keep=4, blocks=8):
    # One backward pass at the unperturbed prompt.  Span proxy is
    # |u_hat| = |-sum_t d(pred_score-gold_score)/d alpha_t|.
    grad = att.grad_alpha(prep)
    score = np.abs(att.u_hat_first_order(prep, spans, g=grad))
    edges = np.linspace(prep.ctx_start, prep.ctx_end, blocks + 1).round().astype(int)
    regions = [(edges[i], edges[i + 1]) for i in range(blocks)
               if edges[i] < edges[i + 1]]
    block_score = np.asarray([
        max((score[i] for i, s in enumerate(spans)
             if s.end > a and s.start < b), default=-np.inf)
        for a, b in regions
    ])
    chosen = np.argsort(-block_score)[:keep]
    return [i for i, s in enumerate(spans)
            if any(s.end > regions[j][0] and s.start < regions[j][1]
                   for j in chosen)]

m.shortlist = gradient_shortlist

def evaluate():
    m.evaluate()
    report = json.loads(m.OUT.read_text())
    report['protocol'] = ('Scientist-known 1084, grouped 3x5 OOF, current127 LR '
                          'unchanged; pure first-order gate-gradient coarse '
                          'screen in both stages; 8 blocks, keep 4')
    # Gradient uses one backward screen per stage, rather than the four
    # attention forwards counted by the inherited query accounting.
    q = []
    for fp in m.CACHE.glob('*.npz'):
        with np.load(fp) as z:
            q.append([int(z['stage1_candidates']), int(z['stage1_full']),
                      int(z['stage2_candidates']), int(z['stage2_full'])])
    q = np.asarray(q)
    report['queries']['screening_note'] = 'two backward passes per item are not represented as forward-equivalent queries'
    report['queries']['fine_perturbation_mean'] = float((q[:, 0] + q[:, 2]).mean())
    report['queries']['fine_perturbation_reduction'] = float(1 - (q[:, 0] + q[:, 2]).sum() / (q[:, 1] + q[:, 3]).sum())
    m.OUT.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['collect', 'evaluate', 'all'])
    p.add_argument('--model', default='NousResearch/Meta-Llama-3.1-8B-Instruct')
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--keep', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    if a.stage in ('collect', 'all'):
        m.collect(a)
    if a.stage in ('evaluate', 'all'):
        evaluate()

if __name__ == '__main__':
    main()

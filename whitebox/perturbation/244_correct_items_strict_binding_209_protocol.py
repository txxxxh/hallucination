#!/usr/bin/env python3
"""Run the exact 209 strict-binding protocol on correct Scientist items.

Everything is inherited from 207/204: whole-prompt two-word windows, measured
u>0 selection, strict attribute matching, and the real/nonce x owner x answer
order assay.  The only necessary adaptation is orienting each correct item as
fixed wrong-vs-right before passing it to the error-only 204 implementation.
"""
from __future__ import annotations
import importlib,sys

base=importlib.import_module('206_scientist_attribute_binding_pilot')
strict=importlib.import_module('207_scientist_strict_attribute_binding_pilot')
span_module=importlib.import_module('125_collect_current_three_benchmarks')
span_module.spans=base.sliding_spans

jobs_module=importlib.import_module('152_scientist_attention_pruned_current127')
original_jobs=jobs_module.jobs

def correct_jobs_as_fixed_wrong_right():
 out=[]
 for key,group,label,prompt,pred,other in original_jobs():
  if not label or pred in ('','None','null') or other in ('','None','null'):
   continue
  # On a correct item, pred is the right answer and other is the fixed wrong
  # answer.  204 expects (pred=wrong, other=right) and only accepts label=0.
  out.append((key,group,0,prompt,other,pred))
 return out

jobs_module.jobs=correct_jobs_as_fixed_wrong_right
experiment=importlib.import_module('204_scientist_binding_override_pilot')
experiment.facts_by_item=base.full_profile_attributes
experiment.match_fact=strict.strict_match

if __name__=='__main__':
 if '--out' not in sys.argv:
  sys.argv.extend(['--out',str(experiment.RUNS/'244_correct_items_strict_binding_209_protocol')])
 experiment.main()

#!/usr/bin/env python3
"""Run decoy generation after disabling the incompatible native BMM override."""
import os
if os.environ.get('SPANATTR_DISABLE_NATIVE_BMM')=='1':
 from torch._native.registry import deregister_op_overrides
 deregister_op_overrides(disable_op_symbols='bmm')
import runpy
if __name__=='__main__':runpy.run_path('131_hotpot_decoys_balance.py',run_name='__main__')

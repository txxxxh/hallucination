#!/usr/bin/env python3
"""Run HotpotQA generation after disabling the incompatible native BMM override."""
import os
if os.environ.get('SPANATTR_DISABLE_NATIVE_BMM')=='1':
 from torch._native.registry import deregister_op_overrides
 deregister_op_overrides(disable_op_symbols='bmm')
import importlib,sys
if __name__=='__main__':
 sys.argv[1:1]=['generate']
 importlib.import_module('130_prepare_hotpotqa').main()

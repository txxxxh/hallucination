#!/usr/bin/env python3
"""Layer-12 full/radial/tangent entity-direction known/unknown pilot."""
import argparse,importlib,json,os,tempfile
from pathlib import Path
import numpy as np
B=importlib.import_module("160_symmetric_evidence_known_unknown");G=importlib.import_module("173_known_unknown_margin_geometry");P=importlib.import_module("163_pics_keen_known_unknown");RUNS=Path(__file__).resolve().parent/"runs";OUT=RUNS/"178_layer12_entity_direction_n50";ALPHA=np.linspace(0,1,5,dtype=np.float32)
def save(path,**v):
 fd,tmp=tempfile.mkstemp(prefix=path.name+".",suffix=".npz",dir=path.parent);os.close(fd)
 try:np.savez_compressed(tmp,**v);os.replace(tmp,path)
 finally:
  if os.path.exists(tmp):os.unlink(tmp)
def score(model,emb,E,ans,layer,ix,target,alpha,mode):
 import torch
 def hook(_m,_i,out):
  h=out[0]if isinstance(out,tuple)else out;z=h.clone();cur=z[:,ix];d=target(h)-cur;rad=(d*cur).sum(-1,keepdim=True)/(cur.square().sum(-1,keepdim=True)+1e-8)*cur
  use=d if mode=="full"else(rad if mode=="radial"else d-rad);z[:,ix]=cur+alpha*use;return(z,*out[1:])if isinstance(out,tuple)else z
 handle=layer.register_forward_hook(hook)
 try:return G.score(model,emb,E,ans)
 finally:handle.remove()
def curves(model,tok,row):
 import torch
 prompt,a,b=G.prompt_for(row);text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt",return_offsets_mapping=True,add_special_tokens=False);ids=enc.input_ids.cuda();off=enc.offset_mapping[0].tolist();a0=text.find(a);b0=text.find(b);ai=G.span(off,a0,a0+len(a));bi=G.span(off,b0,b0+len(b));aid=tok(a,add_special_tokens=False,return_tensors="pt").input_ids[0].cuda();bid=tok(b,add_special_tokens=False,return_tensors="pt").input_ids[0].cuda();emb=model.get_input_embeddings();E=emb(ids).detach();layer=model.model.layers[12];out={}
 for mode in("full","radial","tangent"):
  arms=[]
  for ix,other in((ai,bi),(bi,ai)):
   target=lambda h,other=other:h[:,other].mean(1,keepdim=True).expand(-1,len(ix),-1);vals=[]
   for alpha in ALPHA:
    with torch.inference_mode():vals.append(float((score(model,emb,E,aid,layer,ix,target,float(alpha),mode)-score(model,emb,E,bid,layer,ix,target,float(alpha),mode))[0]))
   arms.append(vals)
  C=np.asarray(arms,np.float32);d=np.abs(C-C[:,:1]);out[mode]=np.r_[np.sort(d.max(1)),np.sort(d.mean(1)),np.sort(np.max(abs(np.diff(C)),1)),np.mean(abs((C[0]-C[0,0])+(C[1]-C[1,0]))),abs(C[0,0])].astype(np.float32)
 return out
def selected(n):
 rows,*_=B.load_rows();used={x["key"]for x in B.select_balanced(rows,100,B.SEED)};second=B.select_balanced([x for x in rows if x["key"]not in used],100,B.SEED);used|={x["key"]for x in second};return B.select_balanced([x for x in rows if x["key"]not in used],n,B.SEED)
def collect(a,rows):
 import torch
 from transformers import AutoModelForCausalLM,AutoTokenizer
 (a.output_dir/"features").mkdir(parents=True,exist_ok=True);tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation="eager",local_files_only=True).cuda().eval()
 for i,r in enumerate(rows,1):
  fp=a.output_dir/"features"/(r["key"]+".npz")
  if a.resume and fp.exists():continue
  try:save(fp,**curves(model,tok,r));print(f"[{i}/{len(rows)}] {r['key']}",flush=True)
  except Exception as e:B.append_error(a.output_dir/"errors.jsonl",{"key":r["key"],"error":repr(e)});print("ERROR",r["key"],repr(e),flush=True)
def evaluate(a,rows):
 E=importlib.import_module("175_eval_margin_geometry_nested");rows=[r for r in rows if(a.output_dir/"features"/(r["key"]+".npz")).exists()];y=np.array([r["known"]for r in rows]);Q=np.stack([np.load(B.QUESTION_CACHE/(r["key"]+".npz"))["hidden"][P.KEEN_LAYERS].astype(np.float32)for r in rows]);report={"n":len(y),"known":int(y.sum()),"layer_zero_based":12,"alphas":ALPHA.tolist(),"results":{}}
 for mode in("full","radial","tangent"):
  X=np.stack([np.load(a.output_dir/"features"/(r["key"]+".npz"))[mode]for r in rows]);runs=[E.run_head(Q,X,y,s)for s in E.SEEDS];report["results"][mode]={"feature_only":{"mean":E.met(y,np.mean([z[1]for z in runs],0)),"per_seed":[E.met(y,z[1])for z in runs]},"question_only":{"mean":E.met(y,np.mean([z[0]for z in runs],0))},"nested_fusion":{"mean":E.met(y,np.mean([z[2]for z in runs],0)),"per_seed":[E.met(y,z[2])for z in runs]}}
 B.atomic_json(a.output_dir/"evaluation.json",report);print(json.dumps(report,indent=2))
def main():
 p=argparse.ArgumentParser();p.add_argument("stage",choices=("collect","evaluate","all"));p.add_argument("--limit",type=int,default=50);p.add_argument("--model",default="NousResearch/Meta-Llama-3.1-8B-Instruct");p.add_argument("--output-dir",type=Path,default=OUT);p.add_argument("--resume",action="store_true");a=p.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);rows=selected(a.limit)
 if a.stage in("collect","all"):collect(a,rows)
 if a.stage in("evaluate","all"):evaluate(a,rows)
if __name__=="__main__":main()

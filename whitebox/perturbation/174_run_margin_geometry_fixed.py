#!/usr/bin/env python3
"""Runner for 173 with the BF16 interpolation-path correction."""
import importlib,numpy as np,zlib
B=importlib.import_module("173_known_unknown_margin_geometry")
def one(model,tok,r,draws):
 import torch
 prompt,a,b=B.prompt_for(r);text=tok.apply_chat_template([{"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt",return_offsets_mapping=True,add_special_tokens=False);ids=enc.input_ids.cuda();off=enc.offset_mapping[0].tolist();q0=text.find(r["question"]);qi=B.span(off,q0,q0+len(r["question"]));a0=text.find(a);b0=text.find(b);ai=B.span(off,a0,a0+len(a));bi=B.span(off,b0,b0+len(b));aid=tok(a,add_special_tokens=False,return_tensors="pt").input_ids[0].cuda();bid=tok(b,add_special_tokens=False,return_tensors="pt").input_ids[0].cuda();emb=model.get_input_embeddings();E=emb(ids).detach().requires_grad_(True);m=B.margin(model,emb,E,aid,bid).sum();G,=torch.autograd.grad(m,E);g=G[0].float();ebf=E[0].detach();e=ebf.float();qg=g[qi];mean=emb.weight.detach().float().mean(0);exact=np.r_[abs(float(m.detach())),B.conc(qg.norm(dim=-1).cpu()),B.conc(abs((qg*e[qi]).sum(-1)).cpu()),B.conc(abs((qg*(mean-e[qi])).sum(-1)).cpu()),B.conc(g[ai].norm(dim=-1).cpu()),B.conc(g[bi].norm(dim=-1).cpu())].astype(np.float32);gen=torch.Generator(device="cuda");gen.manual_seed(B.BASE.SEED+zlib.crc32(r["key"].encode()));rp=[]
 for _ in range(draws):
  n=torch.randn(qg.shape,generator=gen,device=qg.device);n=n/(n.norm(dim=-1,keepdim=True)+1e-8)*e[qi].norm(dim=-1,keepdim=True);rp.append(float((qg*n).sum()))
 rp=np.asarray(rp,np.float32);rnd=np.asarray([abs(float(m.detach())),np.mean(abs(rp)),np.std(rp),np.max(abs(rp)),np.mean(rp),np.std(abs(rp)),np.mean(rp>0)],np.float32);C=[]
 with torch.inference_mode():
  for ix,target in((ai,ebf[bi].mean(0)),(bi,ebf[ai].mean(0))):
   vals=[]
   for alpha in B.ALPHAS:
    ep=ebf.clone()[None];ep[:,ix]=(1-float(alpha))*ep[:,ix]+float(alpha)*target;vals.append(float(B.margin(model,emb,ep,aid,bid)[0]))
   C.append(vals)
 C=np.asarray(C,np.float32);return exact,rnd,B.curvefeat(C),C
B.one=one
if __name__=="__main__":B.main()

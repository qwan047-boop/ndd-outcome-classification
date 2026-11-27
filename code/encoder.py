"""Historical chunking and pooling. Loads only supplied local encoder files."""
import os
import numpy as np
from sklearn.preprocessing import normalize

def encode(texts, modelpath, log=print):
 len_all = len(texts)
 import torch,sentence_transformers,transformers
 from sentence_transformers import SentenceTransformer
 torch.set_num_threads(min(4,os.cpu_count() or 1))
 model=SentenceTransformer(str(modelpath),local_files_only=True,trust_remote_code=False,device='cpu')
 tok=model.tokenizer;limit=model.max_seq_length;step=limit-tok.num_special_tokens_to_add(pair=False)
 chunks=[];owners=[];weights=[];lengths=[]
 for j,t in enumerate(texts):
  ids=tok(t,add_special_tokens=False,truncation=False)['input_ids'];lengths.append(len(ids))
  for k in range(0,max(1,len(ids)),step):
   tt=ids[k:k+step]
   assert tok.num_special_tokens_to_add(pair=False)==2 and tok.cls_token_id is not None and tok.sep_token_id is not None
   # Explicit standard single-sequence BERT framing, verified against tokenizer below.
   data={'input_ids':[tok.cls_token_id]+tt+[tok.sep_token_id],'attention_mask':[1]*(len(tt)+2),'token_type_ids':[0]*(len(tt)+2)}
   if k==0 and len(ids)<=step:assert tok(t,truncation=False)['input_ids']==data['input_ids']
   assert len(data['input_ids'])<=limit
   chunks.append(data);owners.append(j);weights.append(max(1,len(tt)))
 # Direct tensor feature batches preserve exact token IDs; no decode/re-tokenize or truncation.
 E=np.zeros((len_all,model.get_sentence_embedding_dimension()),dtype=np.float32);total=np.zeros(len_all)
 for k in range(0,len(chunks),32):
  cc=chunks[k:k+32];n=max(len(x['input_ids']) for x in cc)
  batch={key:torch.tensor([x[key]+([tok.pad_token_id] if key=='input_ids' else [0])*(n-len(x[key])) for x in cc],dtype=torch.long) for key in ['input_ids','attention_mask','token_type_ids']}
  with torch.inference_mode():emb=model(batch)['sentence_embedding'];emb=torch.nn.functional.normalize(emb,p=2,dim=1).cpu().numpy()
  for ii,z in enumerate(emb,k):E[owners[ii]]+=z*weights[ii];total[owners[ii]]+=weights[ii]
  if k%320==0:log(f'ENCODE chunks {k}/{len(chunks)}')
 X=normalize(E/total[:,None]).astype(np.float32)
 return X

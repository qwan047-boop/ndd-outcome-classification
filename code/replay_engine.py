"""Five-domain real-data offline replay; fit/query functions never accept test gold."""
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
from pathlib import Path
import sys,json,csv,hashlib,time,platform,collections,warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics import f1_score,precision_recall_fscore_support,accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import sklearn,scipy
ROOT=Path(__file__).resolve().parents[1]
R=ROOT
O=Path(sys.argv[1]);backend=sys.argv[2]
cfg=json.loads((O/'EXPERIMENT_CONFIG.yaml').read_text(encoding='utf8'));L=cfg['labels']
def load(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def log(t):
 print(t,flush=True)
 with (O/'execution.log').open('a',encoding='utf8') as f:f.write(time.strftime('%Y-%m-%d %H:%M:%S')+' '+t+'\n')
def save(n,obj):(O/n).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
parts=load(R/'data/locked_partitions.json');rows=load(R/'data/pediatric_outcomes.json');by={x['outcome_id']:x for x in rows}
trainids=parts['training_ids'];testids=parts['test_ids'];poolids=parts['pool_ids'];allids=trainids+testids;loc={i:j for j,i in enumerate(allids)}
assert not set(trainids)&set(testids)
for k in ['nct','strict_split_group','normalized_hash']:assert not {by[i][k] for i in trainids}&{by[i][k] for i in testids}
weak={x['record_id']:x['rule_labels'] for x in load(R/'data/training_weak_labels.json') if x['rule_status']=='Rule_candidate'}
assert len(weak)==1934 and set(weak)<=set(trainids)
assert sha(O/'HUMAN_REFERENCE_FINAL_FROZEN.csv')==cfg['human_reference_sha256']
with (O/'HUMAN_REFERENCE_FINAL_FROZEN.csv').open(encoding='utf-8-sig') as f:human=list(csv.DictReader(f))
# Pool-only oracle. The trainer does not receive a test-gold argument.
oracle_data={x['Record_ID']:x['Current_Author_Label'].split(';') for x in human if x['Partition']=='Pool_40'}
assert set(oracle_data)==set(poolids)
def encode(labs):return np.array([int(x in labs) for x in L],dtype=np.int8)
class Oracle:
 def __init__(self):self.released=[]
 def reveal(self,ids):
  assert len(ids)==len(set(ids)) and not set(ids)&set(self.released) and set(ids)<=set(poolids)
  self.released.extend(ids);return {i:oracle_data[i] for i in ids}
def fit_predict(X,targets,seed):
 assert targets and set(targets)<=set(trainids) and not set(targets)&set(testids)
 ids=sorted(targets);Y=np.stack([encode(targets[i]) for i in ids]);xx=X[[loc[i] for i in ids]]
 models=[]
 for c in range(len(L)):
  vals=np.unique(Y[:,c])
  if len(vals)==1:models.append(int(vals[0]));continue
  est=LinearSVC(C=1,class_weight='balanced',random_state=seed) if backend=='tfidf_svm' else LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=seed)
  with warnings.catch_warnings(record=True) as warns:
   est.fit(xx,Y[:,c]);assert not any('converg' in str(w.message).lower() for w in warns),'Convergence warning: stop, do not silently change params'
  models.append(est)
 def scores(ids):
  xx=X[[loc[i] for i in ids]];ss=[]
  for est in models:
   if isinstance(est,int):ss.append(np.full(len(ids),1.0 if est else -1.0))
   elif backend=='tfidf_svm':ss.append(est.decision_function(xx))
   else:ss.append(2*est.predict_proba(xx)[:,1]-1)
  return np.stack(ss,axis=1)
 return scores,{'training_n':len(ids),'positive_support':Y.sum(axis=0).tolist(),'constant_classes':[L[c] for c in range(len(L)) if len(np.unique(Y[:,c]))==1]}
def choose(scores,X,ids,method,rng):
 if method=='random':return [ids[j] for j in rng.choice(len(ids),10,replace=False)]
 rank=np.argsort(np.abs(scores).mean(axis=1),kind='stable')
 if method=='uncertainty':return [ids[j] for j in rank[:10]]
 cand=list(rank[:min(len(ids),50)]);selected=[cand.pop(0)]
 while len(selected)<10:
  a=X[[loc[ids[j]] for j in cand]];b=X[[loc[ids[j]] for j in selected]]
  dist=1-cosine_similarity(a,b).max(axis=1);selected.append(cand.pop(int(np.argmax(dist))))
 return [ids[j] for j in selected]

texts=[by[i]['measure']+'\n'+by[i]['description'] for i in allids]
start=time.perf_counter();log('START '+backend)
if backend=='tfidf_svm':
 vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True)
 vec.fit(texts[:len(trainids)]);X=vec.transform(texts)
 save('TFIDF_FEATURE_PROVENANCE.json',{'fit_ids_sha256':hashlib.sha256('\n'.join(trainids).encode()).hexdigest(),'fit_rows':len(trainids),'vocabulary_n':len(vec.vocabulary_),'no_test_fit':True,'pool_unlabeled_text_in_vocabulary':True})
elif backend=='minilm_lr':
 X=np.load(ROOT/'history/baseline/MINILM_EMBEDDINGS.npy',allow_pickle=False)
 assert load(ROOT/'history/baseline/MINILM_FEATURE_PROVENANCE.json')['ids']==allids
else:raise ValueError(backend)

predictions=[];queries=[];snapshots=[]
for seed in cfg['seeds']:
 for strategy in cfg['strategies']:
  oracle=Oracle();targets={i:list(v) for i,v in weak.items()};released_labels={};rng=np.random.default_rng(seed)
  for budget in cfg['budgets']:
   assert len(oracle.released)==budget
   scorer,meta=fit_predict(X,targets,seed)
   scores=scorer(testids);pred=(scores>0).astype(int)
   human_usable=sum(v!=['UNCLEAR'] for v in released_labels.values())
   meta.update({'backend':backend,'strategy':strategy,'seed':seed,'budget':budget,'human_reviewed_n':budget,'human_trainable_n':human_usable,'weak_remaining_n':sum(i not in released_labels for i in targets)})
   snapshots.append(meta)
   for rid,pr,sc in zip(testids,pred,scores):predictions.append({'model':backend,'strategy':strategy,'seed':seed,'budget':budget,'Record_ID':rid,'predicted_labels':[x for x,y in zip(L,pr) if y],'scores':sc.tolist(),'training_n':meta['training_n'],'human_trainable_n':human_usable})
   if budget==40:break
   unseen=[i for i in poolids if i not in oracle.released]
   selected=choose(scorer(unseen),X,unseen,strategy,rng)
   # Only now access the pool's selected gold labels.
   revealed=oracle.reveal(selected);released_labels.update(revealed)
   for rid,labs in revealed.items():
    if labs==['UNCLEAR']:targets.pop(rid,None)
    else:targets[rid]=labs
   queries.append({'backend':backend,'strategy':strategy,'seed':seed,'from_budget':budget,'to_budget':budget+10,'selected_ids':selected,'revealed_labels':revealed})
  log(f'REPLAY {backend} seed={seed} strategy={strategy} complete')
# Independent assertions: matched data at b0, all Pool released at b40, no test release.
for q in queries:assert not set(q['selected_ids'])&set(testids)
assert len(predictions)==len(cfg['seeds'])*len(cfg['strategies'])*len(cfg['budgets'])*60
save(backend+'_PREDICTIONS.json',predictions);save(backend+'_QUERY_RELEASE_LOG.json',queries);save(backend+'_TRAINING_SNAPSHOTS.json',snapshots)
save(backend+'_EXECUTION.json',{'elapsed_seconds':time.perf_counter()-start,'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__,'fit_count':len(snapshots),'test_labels_used_by_fitting':False,'input_config_sha256':sha(O/'EXPERIMENT_CONFIG.yaml'),'gold_sha256':sha(O/'HUMAN_REFERENCE_FINAL_FROZEN.csv')})
log('FINISHED '+backend)

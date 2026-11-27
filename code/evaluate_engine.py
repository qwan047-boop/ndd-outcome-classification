"""Evaluate saved predictions after all fit/query paths; no fitting or tuning."""
from pathlib import Path
import sys,json,csv,hashlib,collections,ast,re
import numpy as np
from sklearn.metrics import f1_score,precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[1];O=Path(sys.argv[1]);R=ROOT
load=lambda p:json.loads(p.read_text(encoding='utf8'))
cfg=load(O/'EXPERIMENT_CONFIG.yaml');L=cfg['labels'];parts=load(R/'data/locked_partitions.json');testids=parts['test_ids']
with (O/'HUMAN_REFERENCE_FINAL_FROZEN.csv').open(encoding='utf-8-sig') as f:human={x['Record_ID']:x for x in csv.DictReader(f)}
gold={i:human[i]['Current_Author_Label'].split(';') for i in testids};valid=[i for i in testids if gold[i]!=['UNCLEAR']]
encode=lambda labs:np.array([int(x in labs) for x in L],dtype=int)
Y=np.stack([encode(gold[i]) for i in valid]);assert len(valid)==58
source_path=R/'data/pediatric_outcomes.json'
source_available=source_path.exists()
by=({x['outcome_id']:x for x in load(source_path)} if source_available else
    {i:{'nct':i.split(':')[0],'measure':'WITHHELD_SOURCE_TEXT','description':'WITHHELD_SOURCE_TEXT'} for i in testids})
(O/'METRIC_SCOPE.json').write_text(json.dumps({
    'evaluation':'ARCHIVED_PREDICTION_RESCORING',
    'source_text_available':source_available,
    'rule_baseline':'REEXECUTED_ON_SOURCE' if source_available else 'SAVED_RULE_PREDICTIONS',
    'rule_extraction_revalidated':source_available,
    'model_refitting':False
},indent=2),encoding='utf8')
def savecsv(n,rr):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
def numeric(y,p):
 tp=(y*p).sum(axis=0);fp=((1-y)*p).sum(axis=0);fn=(y*(1-p)).sum(axis=0)
 f=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(5,dtype=float),where=2*tp+fp+fn!=0)
 denom=2*tp.sum()+fp.sum()+fn.sum()
 return {'micro_f1':float(2*tp.sum()/denom) if denom else 0.,'macro_f1':float(f.mean()),'exact_match':float(np.all(y==p,axis=1).mean())}
boot=np.random.default_rng(20261123).integers(0,len(valid),size=(1000,len(valid)))
def ci(p):
 y=Y[boot];pp=p[boot];tp=(y*pp).sum(axis=1);fp=((1-y)*pp).sum(axis=1);fn=(y*(1-pp)).sum(axis=1);den=2*tp+fp+fn
 macro=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean(axis=1)
 dd=den.sum(axis=1);micro=np.divide(2*tp.sum(axis=1),dd,out=np.zeros_like(dd,dtype=float),where=dd!=0)
 return dict(zip(['micro_ci_low','micro_ci_high','macro_ci_low','macro_ci_high'],[float(z) for a in [micro,macro] for z in np.quantile(a,[.025,.975])]))
# Reconstruct pre-existing frozen rules without changing them to match human corrections.
tree=ast.parse((R/'code/archived_rule_definitions.py').read_text(encoding='utf8'))
main=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main')
assigns={x.targets[0].id:x.value for x in main.body if isinstance(x,ast.Assign) and isinstance(x.targets[0],ast.Name)}
taxonomy=ast.literal_eval(assigns['labels']);instruments=ast.literal_eval(assigns['instruments']);mapping={x:k for k,v in taxonomy.items() for x in v}
rules=load(ROOT/'03_annotation/label_rules.json')
def frozen_rule(row):
 text=row['measure']+'\n'+row['description'];fam=[]
 for conf in [rules,instruments]:fam.append({mapping[lab] for lab,pat in conf.items() if re.search(pat,text,re.I)})
 labs=fam[0]|fam[1];conflict=bool(fam[0] and fam[1] and fam[0]!=fam[1])
 return sorted(labs),conflict
if source_available:
 weak=load(R/'data/training_weak_labels.json')
 for x in weak:
  labs,conflict=frozen_rule(by[x['record_id']]);assert labs==x['rule_labels'] and conflict==(x['rule_status']=='Rule_sources_disagree')
 rulepred=[]
 for i in testids:
  labs,conflict=frozen_rule(by[i]);rulepred.append({'model':'rule_baseline','strategy':'none','seed':20261123,'budget':0,'Record_ID':i,'predicted_labels':[] if conflict else labs,'scores':None,'training_n':0,'human_trainable_n':0})
else:
 with (R/'history/baseline/TEST_PREDICTIONS_ALL.csv').open(encoding='utf-8-sig',newline='') as f:
  stored=[x for x in csv.DictReader(f) if x['model']=='rule_baseline' and int(x['budget'])==0]
 assert len(stored)==len({x['Record_ID'] for x in stored})==60
 assert {x['Record_ID'] for x in stored}==set(testids)
 rulepred=[dict(model='rule_baseline',strategy='none',seed=20261123,budget=0,Record_ID=x['Record_ID'],
                predicted_labels=x['predicted_labels'].split(';') if x['predicted_labels'] else [],scores=None,training_n=0,human_trainable_n=0) for x in stored]
allpred=rulepred[:]
for model in ['tfidf_svm','minilm_lr']:
 p=O/(model+'_PREDICTIONS.json')
 if p.exists():allpred+=load(p)
groups=collections.defaultdict(list)
for x in allpred:groups[(x['model'],x['strategy'],x['seed'],x['budget'])].append(x)
metrics=[];classwise=[];flat=[];errors=[];cache={}
for key,rr in sorted(groups.items()):
 assert len(rr)==60 and {x['Record_ID'] for x in rr}==set(testids)
 pred={x['Record_ID']:x['predicted_labels'] for x in rr};P=np.stack([encode(pred[i]) for i in valid]);cache[key]=P
 result=dict(zip(['model','strategy','seed','budget'],key));result.update(numeric(Y,P));result.update(ci(P))
 assert abs(result['micro_f1']-f1_score(Y,P,average='micro',zero_division=0))<1e-12
 assert abs(result['macro_f1']-f1_score(Y,P,average='macro',zero_division=0))<1e-12
 result.update({'test_total':60,'reference_decidable_n':58,'reference_coverage':58/60,'prediction_label_coverage_all60':sum(bool(pred[i]) for i in testids)/60,'abstention_rate_all60':sum(not pred[i] for i in testids)/60,'training_n':rr[0]['training_n'],'human_trainable_n':rr[0]['human_trainable_n']});metrics.append(result)
 precision,recall,f1,support=precision_recall_fscore_support(Y,P,average=None,zero_division=0)
 for j,label in enumerate(L):classwise.append(dict(zip(['model','strategy','seed','budget'],key),label=label,precision=float(precision[j]),recall=float(recall[j]),f1=float(f1[j]),support=int(support[j]),interpretation='exploratory_sparse_support' if support[j]<10 else 'small_single_reviewer_test'))
 for x in rr:
  rid=x['Record_ID'];record={'model':key[0],'strategy':key[1],'seed':key[2],'budget':key[3],'Record_ID':rid,'NCT_ID':by[rid]['nct'],'human_reference':';'.join(gold[rid]),'predicted_labels':';'.join(pred[rid]),'eligible_for_five_label_metrics':rid in valid}
  flat.append(record)
  if key[2]==20261123 and key[3] in [0,40] and (rid not in valid or set(gold[rid])!=set(pred[rid])):
   errors.append(dict(record,Outcome_Measure=by[rid]['measure'],Outcome_Description=by[rid]['description'],error_type='reference_UNCLEAR_not_scored' if rid not in valid else 'label_set_mismatch',manual_error_cause='NOT_ADJUDICATED'))
savecsv('BASELINE_RESULTS.csv',[x for x in metrics if x['budget']==0 and (x['model']=='rule_baseline' or (x['strategy']=='random' and x['seed']==20261123))])
savecsv('ACTIVE_LEARNING_RESULTS.csv',[x for x in metrics if x['model']!='rule_baseline']);savecsv('CLASS_WISE_METRICS.csv',classwise);savecsv('TEST_PREDICTIONS_ALL.csv',flat);savecsv('ERROR_ANALYSIS.csv',errors)
savecsv('LEARNING_CURVE_DATA.csv',[dict(model=m,strategy=s,budget=b,n_seeds=len(v),micro_mean=float(np.mean([x['micro_f1'] for x in v])),micro_min=min(x['micro_f1'] for x in v),micro_max=max(x['micro_f1'] for x in v),micro_sd=float(np.std([x['micro_f1'] for x in v],ddof=1)),macro_mean=float(np.mean([x['macro_f1'] for x in v])),macro_min=min(x['macro_f1'] for x in v),macro_max=max(x['macro_f1'] for x in v),macro_sd=float(np.std([x['macro_f1'] for x in v],ddof=1))) for m in sorted({x['model'] for x in metrics if x['model']!='rule_baseline'}) for s in cfg['strategies'] for b in cfg['budgets'] for v in [[x for x in metrics if x['model']==m and x['strategy']==s and x['budget']==b]]])
assertions={}
for model in ['tfidf_svm','minilm_lr']:
 snaps=O/(model+'_TRAINING_SNAPSHOTS.json')
 if not snaps.exists():continue
 q=load(O/(model+'_QUERY_RELEASE_LOG.json'));ss=load(snaps)
 for seed in cfg['seeds']:
  complete=[]
  for strat in cfg['strategies']:
   runs=[x for x in q if x['seed']==seed and x['strategy']==strat];released=[i for x in runs for i in x['selected_ids']]
   assert len(released)==40 and set(released)==set(parts['pool_ids']) and len(set(released))==40
   assert not set(released)&set(testids);complete.append(cache[(model,strat,seed,40)])
  assert all(np.array_equal(complete[0],p) for p in complete[1:]),'Budget40 data/order inconsistency'
 assert all(x['human_trainable_n']==39 for x in ss if x['budget']==40)
 assertions[model]={'retrain_snapshots':len(ss),'all_selections_unique_pool_only':True,'budget40_training_data_and_predictions_equal_across_strategies':True,'budget40_human_usable':39,'initial_train_support':next(x['positive_support'] for x in ss if x['budget']==0)}
(O/'REPLAY_INTEGRITY_CHECK.json').write_text(json.dumps(assertions,indent=2),encoding='utf8')
print(json.dumps({'baseline':[x for x in metrics if x['budget']==0 and (x['model']=='rule_baseline' or (x['strategy']=='random' and x['seed']==20261123))],'integrity':assertions},ensure_ascii=False))

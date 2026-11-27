"""Bounded final sensitivity/diagnostic analyses; no human or LLM relabeling."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import json,csv,hashlib,collections,re,sys,time,shutil,zipfile
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
ROOT=Path(__file__).resolve().parent;R=ROOT/'Integrated_WeakSupervision_ActiveLearning_v1_20261123';BASE=R/'Formal_5Class_Experiment_20261124T124632';MERGE=R/'LLM_Expansion100_20261124T130714/Provenance_And_Merge_20261124T134746'
O=R/('Final_Optimization_'+datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S'));O.mkdir();(O/'FINAL_FIGURES').mkdir();(O/'evidence').mkdir();(O/'code').mkdir()
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def csvsave(n,rows):
 with (O/n).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def log(s):print(s,flush=True)
parts=load(R/'data/locked_partitions.json');by={x['outcome_id']:x for x in load(R/'data/pediatric_outcomes.json')};weak={x['record_id']:x for x in load(R/'data/training_weak_labels.json')};merged=load(MERGE/'LLM_PREDICTIONS_200_MERGED.json')['records'];L=load(BASE/'EXPERIMENT_CONFIG.yaml')['labels']
trainids=parts['training_ids'];testids=parts['test_ids'];ids=sorted(i for i,x in weak.items() if x['rule_status']=='Rule_candidate');assert len(ids)==1934
C={x['Record_ID'] for x in merged if x['QC_Status']=='NONABSTAIN_WEAK_CANDIDATE_NO_IDENTIFIED_FLAGS'}
conflicts=sorted(x['Record_ID'] for x in merged if x['Record_ID'] in C and x['Record_ID'] in ids and set(x['Rule_Labels'])!=set(x['Original_LLM_Prediction']['Outcome_Domain']))
assert len(conflicts)==10
for k in ['nct','strict_split_group','normalized_hash']:assert not {by[i][k] for i in trainids}&{by[i][k] for i in testids}
config={'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),'analysis_type':'POST_TEST_EXPLORATORY_HISTORICAL_CLOSED_WORLD_SENSITIVITY','seed':20261124,'strategies':{'A_original':{'conflict_weight':1},'B_exclude':{'conflict_weight':0},'C_downweight':{'conflict_weight':.25}},'conflict_definition':'different complete positive label sets on10 rule-nonconflicting and LLM QC-candidate records; not proof of true categorical contradiction','conflict_ids':conflicts,'noisy_target_assumption':'only for this historical sensitivity: unassigned classes encoded0 as assumed weak negatives, never confirmed NEGATIVE; strict three-state audit remains unchanged','CV':'5-fold GroupKFold on strict components; assert NCT isolation; validation references remain original weak labels, NOT human gold','selection':'no hyperparameter or strategy selection; fixed C1,balanced; test used only after config freeze','models':['tfidf_svm','minilm_lr'],'bootstrap':{'replicates':2000,'unit':'NCT, one per test record','paired':True,'seed':20261124,'fixed_five_label_macro_zero_division':0},'LLM_fusion_training':False,'human_relabeling':False}
save('FINAL_ANALYSIS_CONFIG_FROZEN.json',config);log('CONFIG_FROZEN '+str(O))
protected=[BASE/'HUMAN_REFERENCE_FINAL_FROZEN.csv',BASE/'BASELINE_RESULTS.csv',BASE/'ACTIVE_LEARNING_RESULTS.csv',R/'data/locked_partitions.json',R/'data/training_weak_labels.json',MERGE/'LLM_PREDICTIONS_200_MERGED.json'];manifest={str(p):sha(p) for p in protected};save('INPUT_HASH_MANIFEST.json',manifest)
text=lambda i:by[i]['measure']+'\n'+by[i]['description'];enc=lambda labs:np.array([int(x in labs) for x in L],dtype=int)
Y=np.stack([enc(weak[i]['rule_labels']) for i in ids]);folds=list(GroupKFold(n_splits=5).split(ids,groups=[by[i]['strict_split_group'] for i in ids]));foldout=[]
for k,(tr,va) in enumerate(folds,1):
 assert not {by[ids[j]]['nct'] for j in tr}&{by[ids[j]]['nct'] for j in va}
 assert not {by[ids[j]]['strict_split_group'] for j in tr}&{by[ids[j]]['strict_split_group'] for j in va}
 for j in va:foldout.append({'Record_ID':ids[j],'Fold':k,'Strict_Group':by[ids[j]]['strict_split_group']})
csvsave('GROUPED_CV_FOLD_ASSIGNMENTS.csv',foldout)
embedding_info=load(BASE/'MINILM_FEATURE_PROVENANCE.json');E=np.load(BASE/'MINILM_EMBEDDINGS.npy',allow_pickle=False);eloc={i:j for j,i in enumerate(embedding_info['ids'])};assert E.shape[0]==len(eloc)
def fit(X,y,weights,kind):
 assert weights.min()>0
 models=[]
 for c in range(5):
  assert len(np.unique(y[:,c]))==2,'Do not force train absent class'
  m=LinearSVC(C=1,class_weight='balanced',random_state=20261123) if kind=='tfidf_svm' else LogisticRegression(C=1,class_weight='balanced',max_iter=2000,random_state=20261123)
  m.fit(X,y[:,c],sample_weight=weights);models.append(m)
 return models
def scores(models,X,kind):return np.stack([m.decision_function(X) if kind=='tfidf_svm' else 2*m.predict_proba(X)[:,1]-1 for m in models],axis=1)
def met(y,p):return {'micro_f1':float(f1_score(y,p,average='micro',zero_division=0)),'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),'exact_match':float(np.all(y==p,axis=1).mean())}
noise=[];predout={};composition=[]
for kind in ['tfidf_svm','minilm_lr']:
 for fold,(tr,va) in enumerate(folds,1):
  if kind=='tfidf_svm':v=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True);xt=v.fit_transform([text(ids[j]) for j in tr]);xv=v.transform([text(ids[j]) for j in va])
  else:xt=E[[eloc[ids[j]] for j in tr]];xv=E[[eloc[ids[j]] for j in va]]
  for strategy,cfg in config['strategies'].items():
   weights=np.array([cfg['conflict_weight'] if ids[j] in conflicts else 1. for j in tr]);keep=weights>0;m=fit(xt[keep],Y[tr][keep],weights[keep],kind);p=scores(m,xv,kind)>0
   noise.append({'model':kind,'strategy':strategy,'scope':'TRAINING_GROUP_CV_WEAK_REFERENCE_NOT_GOLD','fold':fold,'n_train':int(keep.sum()),'n_evaluated':len(va),'n_conflict_train':sum(ids[j] in conflicts for j in tr),'conflict_weight':cfg['conflict_weight'],**met(Y[va],p)})
  log(kind+' CV fold '+str(fold)+' done')
 if kind=='tfidf_svm':v=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True);v.fit([text(i) for i in trainids]);xt=v.transform([text(i) for i in ids]);xe=v.transform([text(i) for i in testids])
 else:xt=E[[eloc[i] for i in ids]];xe=E[[eloc[i] for i in testids]]
 for strategy,cfg in config['strategies'].items():
  weights=np.array([cfg['conflict_weight'] if i in conflicts else 1. for i in ids]);keep=weights>0;m=fit(xt[keep],Y[keep],weights[keep],kind);sc=scores(m,xe,kind);predout[kind+'__'+strategy]={'ids':testids,'scores':sc.tolist(),'predictions':(sc>0).astype(int).tolist(),'n_train':int(keep.sum())}
  for i,wt in zip(ids,weights):composition.append({'model':kind,'strategy':strategy,'Record_ID':i,'weight':wt,'source_set_disagreement':i in conflicts,'included':wt>0,'reference_type':'RULE_WEAK_MULTI_HOT_CLOSED_WORLD_ASSUMPTION'})
save('NOISE_PREDICTIONS_FROZEN.json',predout);csvsave('NOISE_TRAINING_COMPOSITION.csv',composition)
# Only now access frozen Test reference for post-hoc evaluation, not selection.
with (BASE/'HUMAN_REFERENCE_FINAL_FROZEN.csv').open(encoding='utf-8-sig') as f:human={x['Record_ID']:x['Current_Author_Label'].split(';') for x in csv.DictReader(f) if x['Partition']=='Test_60'}
valid=[i for i in testids if human[i]!=['UNCLEAR']];gold=np.stack([enc(human[i]) for i in valid]);testloc={i:j for j,i in enumerate(testids)};boot=np.random.default_rng(20261124).integers(0,58,size=(2000,58))
def bootmetric(p):
 y=gold[boot];z=p[boot];tp=(y*z).sum(1);fp=((1-y)*z).sum(1);fn=(y*(1-z)).sum(1);den=2*tp+fp+fn
 fs=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0);d=den.sum(1);mi=np.divide(2*tp.sum(1),d,out=np.zeros_like(d,dtype=float),where=d!=0)
 return mi,fs.mean(1)
historical={}
for kind in ['tfidf_svm','minilm_lr']:
 pp=[x for x in load(BASE/(kind+'_PREDICTIONS.json')) if x['seed']==20261123 and x['strategy']=='random' and x['budget']==0];lookup={x['Record_ID']:x for x in pp};historical[kind]=np.stack([enc(lookup[i]['predicted_labels']) for i in valid])
 reproduced=np.array(predout[kind+'__A_original']['predictions'])[[testloc[i] for i in valid]];assert np.array_equal(reproduced,historical[kind]),'Historical reproduction failed; stop before writing claims'
 bm=bootmetric(historical[kind])
 for strategy,cfg in config['strategies'].items():
  out=predout[kind+'__'+strategy];p=np.array(out['predictions'])[[testloc[i] for i in valid]];mb=bootmetric(p);delta=[np.quantile(mb[j]-bm[j],[.025,.975]).tolist() for j in range(2)];base_m=met(gold,historical[kind]);mm=met(gold,p)
  noise.append({'model':kind,'strategy':strategy,'scope':'POST_TEST_EXPLORATORY_HUMAN_TEST','fold':0,'n_train':out['n_train'],'n_evaluated':58,'n_conflict_train':10 if strategy!='B_exclude' else 0,'conflict_weight':cfg['conflict_weight'],**mm,'delta_micro':mm['micro_f1']-base_m['micro_f1'],'delta_macro':mm['macro_f1']-base_m['macro_f1'],'delta_micro_ci_low':delta[0][0],'delta_micro_ci_high':delta[0][1],'delta_macro_ci_low':delta[1][0],'delta_macro_ci_high':delta[1][1]})
# Rectangular CSV including CV rows with deliberately absent human comparison fields.
keys=list(dict.fromkeys(k for x in noise for k in x));csvsave('WEAK_LABEL_NOISE_SENSITIVITY.csv',[{k:x.get(k,'') for k in keys} for x in noise])
sv=historical['tfidf_svm'];mi=historical['minilm_lr'];sok=np.all(sv==gold,axis=1);mok=np.all(mi==gold,axis=1);overlap=collections.Counter('Both correct' if a and b else 'SVM only correct' if a else 'MiniLM only correct' if b else 'Both incorrect' for a,b in zip(sok,mok))
comprec=[]
for j,i in enumerate(valid):
 textfull=text(i);inst=[n for n,pat in [('CGI',r'Clinical Global Impression|\bCGI[- ](?:S|I)\b'),('ABC',r'Aberrant Behavior Checklist|\bABC\b'),('SRS',r'Social Responsiveness|\bSRS'),('Vineland',r'Vineland'),('CARS',r'\bCARS'),('SKAMP',r'\bSKAMP') ] if re.search(pat,textfull,re.I)]
 comprec.append({'Record_ID':i,'NCT_ID':by[i]['nct'],'reference':';'.join(human[i]),'svm_prediction':';'.join(L[c] for c,v in enumerate(sv[j]) if v),'minilm_prediction':';'.join(L[c] for c,v in enumerate(mi[j]) if v),'svm_exact_match':bool(sok[j]),'minilm_exact_match':bool(mok[j]),'overlap_group':'Both correct' if sok[j] and mok[j] else 'SVM only correct' if sok[j] else 'MiniLM only correct' if mok[j] else 'Both incorrect','character_length':len(textfull),'word_length':len(textfull.split()),'instrument_mentions':';'.join(inst) or 'Other_or_unspecified','reference_label_count':len(human[i]),'compound_text_proxy':bool(re.search(r';|\b(?:and|subscales|composite)\b',textfull,re.I)),'Outcome_Measure':by[i]['measure'],'Outcome_Description':by[i]['description']})
csvsave('MODEL_ERROR_OVERLAP.csv',comprec)
classcounts=[]
for kind,p in [('tfidf_svm',sv),('minilm_lr',mi)]:
 for c,label in enumerate(L):
  tp=int((gold[:,c]*p[:,c]).sum());fp=int(((1-gold[:,c])*p[:,c]).sum());fn=int((gold[:,c]*(1-p[:,c])).sum());classcounts.append({'model':kind,'label':label,'TP':tp,'FP':fp,'FN':fn,'support':int(gold[:,c].sum()),'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0})
csvsave('ERROR_CLASS_COUNTS.csv',classcounts)
diff0=met(gold,sv);diff1=met(gold,mi);bm0=bootmetric(sv);bm1=bootmetric(mi);diff={'direction':'SVM minus MiniLM','micro_difference':diff0['micro_f1']-diff1['micro_f1'],'macro_difference':diff0['macro_f1']-diff1['macro_f1'],'micro_CI95':np.quantile(bm0[0]-bm1[0],[.025,.975]).tolist(),'macro_CI95':np.quantile(bm0[1]-bm1[1],[.025,.975]).tolist(),'overlap_counts':dict(overlap),'n_test_decidable':58,'interpretation':'paired exploratory bootstrap; not independent confirmation'};save('COMPLEMENTARITY_SUMMARY.json',diff)
descriptive=[]
for feature in ['svm_exact_match','minilm_exact_match']:
 for flag in [True,False]:
  rr=[x for x in comprec if x[feature]==flag];descriptive.append({'model':feature,'exact_match':flag,'n':len(rr),'median_words':float(np.median([x['word_length'] for x in rr])) if rr else None,'median_reference_labels':float(np.median([x['reference_label_count'] for x in rr])) if rr else None,'compound_proxy_n':sum(x['compound_text_proxy'] for x in rr)})
save('ERROR_DESCRIPTIVE_ASSOCIATIONS.json',descriptive)
instruments=[]
for inst in ['CGI','ABC','SRS','Vineland','CARS','SKAMP']:
 rr=[x for x in comprec if inst in x['instrument_mentions'].split(';')];instruments.append({'instrument':inst,'n':len(rr),'svm_exact_correct':sum(x['svm_exact_match'] for x in rr),'minilm_exact_correct':sum(x['minilm_exact_match'] for x in rr),'interpretation':'mention-based overlapping groups; no reviewed instrument ontology'})
save('INSTRUMENT_ERROR_PATTERNS.json',instruments)
coverage=[]
for cohort in ['original_pilot100','expansion100']:
 rr=[x for x in merged if x['Cohort']==cohort];non=[x for x in rr if not x['Original_LLM_Prediction']['Abstain']];both=[x for x in rr if x['Nonconflicting_Rule_and_Nonabstain_LLM']]
 full=partial=disjoint=0
 for x in both:
  a=set(x['Rule_Labels']);b=set(x['Original_LLM_Prediction']['Outcome_Domain'])
  if a==b:full+=1
  elif a&b:partial+=1
  else:disjoint+=1
 coverage.append({'cohort':cohort,'n':100,'raw_nonabstain':len(non),'abstain':100-len(non),'candidate':sum(x['QC_Status']=='NONABSTAIN_WEAK_CANDIDATE_NO_IDENTIFIED_FLAGS' for x in rr),'semantic_pending':sum(x['QC_Status']=='SEMANTIC_REVIEW_PENDING' for x in rr),'both_usable_rule_LLM':len(both),'full_agreement':full,'partial_agreement':partial,'disjoint_sets':disjoint,'consensus_QC_candidate':sum(x['Conservative_Fusion_Candidate'] for x in rr),'no_rule_nonabstain_LLM':sum(not x['Rule_Labels'] and not x['Original_LLM_Prediction']['Abstain'] for x in rr),'LLM_only_usable_candidate':sum(x['QC_Status']=='NONABSTAIN_WEAK_CANDIDATE_NO_IDENTIFIED_FLAGS' and x['Rule_Status']!='Rule_candidate' for x in rr),'OtherDomain_raw':sum('OtherDomain' in x['Original_LLM_Prediction']['Outcome_Domain'] for x in non)})
csvsave('LLM_COVERAGE_COMPARISON.csv',coverage)
support=[]
for cohort in ['original_pilot100','expansion100']:
 for label in L:
  rr=[x for x in merged if x['Cohort']==cohort];support.append({'cohort':cohort,'label':label,'raw_nonabstain_positive':sum(label in x['Original_LLM_Prediction']['Outcome_Domain'] and not x['Original_LLM_Prediction']['Abstain'] for x in rr),'candidate_positive':sum(label in x['Original_LLM_Prediction']['Outcome_Domain'] and x['QC_Status']=='NONABSTAIN_WEAK_CANDIDATE_NO_IDENTIFIED_FLAGS' for x in rr),'fusion_positive':sum(label in x['Original_LLM_Prediction']['Outcome_Domain'] and x['Conservative_Fusion_Candidate'] for x in rr)})
csvsave('LLM_CLASS_SUPPORT.csv',support)
# Replay audit from existing saved predictions/logs only; no AL reruns.
al={}
for kind in ['tfidf_svm','minilm_lr']:
 pp=load(BASE/(kind+'_PREDICTIONS.json'));qq=load(BASE/(kind+'_QUERY_RELEASE_LOG.json'));ss=load(BASE/(kind+'_TRAINING_SNAPSHOTS.json'));first={};maxdiff=0.;flips=0
 for seed in range(20261123,20261128):
  for st in ['random','uncertainty','uncertainty_diversity']:
   runs=[x for x in qq if x['seed']==seed and x['strategy']==st];released=[]
   for k,z in enumerate(runs):assert z['from_budget']==10*k and z['to_budget']==10*(k+1) and len(z['selected_ids'])==10 and not set(released)&set(z['selected_ids']) and set(z['selected_ids'])==set(z['revealed_labels']);released+=z['selected_ids']
   assert len(set(released))==40 and set(released)==set(parts['pool_ids']);first[f'{seed}:{st}']=runs[0]['selected_ids']
   basepred={x['Record_ID']:x for x in pp if x['seed']==seed and x['strategy']==st and x['budget']==0}
   for x in pp:
    if x['seed']==seed and x['strategy']==st:maxdiff=max(maxdiff,float(np.max(np.abs(np.array(x['scores'])-basepred[x['Record_ID']]['scores']))));flips+=x['predicted_labels']!=basepred[x['Record_ID']]['predicted_labels']
 assert len({tuple(first[f'{seed}:random']) for seed in range(20261123,20261128)})==5
 assert all(len({tuple(first[f'{seed}:{s}']) for s in ['random','uncertainty','uncertainty_diversity']})==3 for seed in range(20261123,20261128))
 al[kind]={'snapshots':len(ss),'selected_pool_only_no_duplicates':True,'random_first_batches_distinct_seeds':5,'three_strategies_first_batches_differ_each_seed':True,'max_score_change_all_seeds':maxdiff,'hard_prediction_changes':flips,'baseline_n':ss[0]['training_n'],'final_n':next(x['training_n'] for x in ss if x['budget']==40),'final_human_usable':39,'oracle_limit':'code-path data-flow checked, not OS-level access control; full reference file was opened but fitting and query functions only used pool entries after reveal'}
save('ACTIVE_LEARNING_AUDIT.json',al)
assert all(sha(Path(p))==h for p,h in manifest.items());save('FINAL_ANALYSIS_SUMMARY.json',{'complementarity':diff,'coverage':coverage,'noise_test':[x for x in noise if x['scope']=='POST_TEST_EXPLORATORY_HUMAN_TEST'],'AL':al,'no_new_LLM_labels':True,'strict_three_state_fusion_training_not_run':True,'CV_fits':30,'full_sensitivity_fits':6})
shutil.copy2(Path(__file__),O/'code/final_optimization_analysis.py');log('ANALYSES_COMPLETE '+str(O))

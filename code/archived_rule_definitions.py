"""Frozen-data pediatric audit and blinded annotation preparation; no model accuracy claims."""
from pathlib import Path
from collections import Counter,defaultdict
from fractions import Fraction
import json,re,hashlib,csv,random,shutil,datetime
import openpyxl
P=Path(__file__).resolve().parents[2];ROOT=P.parent
O=P/'Integrated_WeakSupervision_ActiveLearning_v1_20261123'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def age_days(s):
 if not isinstance(s,str):return None
 m=re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*(Years?|Months?|Weeks?|Days?|Hours?|Minutes?)\s*',s,re.I)
 if not m:return None
 v=Fraction(m[1]);u=m[2].lower().rstrip('s')
 return v*{'year':Fraction(36525,100),'month':Fraction(36525,1200),'week':7,'day':1,'hour':Fraction(1,24),'minute':Fraction(1,1440)}[u]
def age_status(lo,hi):
 a,b=age_days(lo),age_days(hi)
 if b is None:return 'Age_pending','Maximum age missing, open-ended or unparseable'
 if a is None:return 'Age_pending','Minimum age missing or unparseable'
 if a<0 or a>b:return 'Age_pending','Age bounds inconsistent'
 if b<Fraction(18)*Fraction(36525,100):return 'Strict_pediatric','Explicit maximum age <18 years; valid minimum'
 return ('Adult_only' if a>=Fraction(18)*Fraction(36525,100) else 'Mixed_or_adult_eligible'),'Maximum age >=18 years'
def load_sheet(p,s):
 w=openpyxl.load_workbook(p,read_only=True,data_only=True);it=iter(w[s].values);keys=next(it);r=[dict(zip(keys,x)) for x in it if x[0]];w.close();return r
def dump(name,obj):
 p=O/name;assert not p.exists(),p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf8')
def main():
 assert not O.exists(),'Refuse overwrite';O.mkdir();(O/'audit').mkdir();(O/'code').mkdir();(O/'data').mkdir();(O/'sources').mkdir()
 stamp=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
 R=json.loads((P/'02_processed_data/outcomes.json').read_text(encoding='utf8'));byid={r['outcome_id']:r for r in R};assert len(byid)==4653
 stage5=ROOT/'02_Screening/stage5_final_20261106_140210/Stage5_Final_Included.xlsx';T=load_sheet(stage5,'Included');assert len(T)==670
 stage6=ROOT/'03_Cleaned_Data/stage6_final_20261106_142449/Stage6_Drug_Classification_Final.xlsx';I=load_sheet(stage6,'Intervention_Final');roles=defaultdict(list)
 for i in I:roles[i['NCT_ID']].append(i)
 inputs=[P/'02_processed_data/outcomes.json',P/'02_processed_data/near_duplicate_pairs.csv',P/'03_annotation/label_rules.json',P/'ml_pipeline.py',P/'active_experiment.py',stage5,stage6,P/'03_annotation/workbook_payload.json']
 manifest={str(p):sha(p) for p in inputs};cache={};study={};audit=[];rawout={};ageconf=[]
 for t in T:
  p=Path(t['Frozen_JSON_File'])
  if str(p) not in cache:
   h=sha(p);assert h==t['Source_SHA256'],f'Frozen hash mismatch {p}';manifest[str(p)]=h;cache[str(p)]=json.loads(p.read_text(encoding='utf8'))
  assert manifest[str(p)]==t['Source_SHA256']
  idx=int(re.search(r'\[(\d+)\]',t['JSON_Record_Path']).group(1));ps=cache[str(p)]['studies'][idx]['protocolSection'];n=t['NCT_ID'];assert ps['identificationModule']['nctId']==n
  study[n]=ps;e=ps.get('eligibilityModule',{});lo=e.get('minimumAge');hi=e.get('maximumAge');ast,reason=age_status(lo,hi)
  if hi!=t['Maximum_Age_Original'] or lo!=t['Minimum_Age_Original']:ageconf.append(n)
  interventions=ps.get('armsInterventionsModule',{}).get('interventions',[]);drugs=[x for x in interventions if x.get('type')=='DRUG']
  # Nonplacebo drug-intervention evidence: active study drug OR randomized active comparator.
  active=[i for i in roles[n] if i['Original_Intervention_Type']=='DRUG' and i['Drug_Role'] in ['Investigational_Active_Drug','Active_Comparator']]
  st=ps.get('designModule',{}).get('studyType');ds='Supported_active_drug' if active else ('Drug_role_pending' if drugs else 'No_DRUG_field')
  status='Eligible_structured' if ast=='Strict_pediatric' and st=='INTERVENTIONAL' and active else ('Excluded_age' if ast in ['Mixed_or_adult_eligible','Adult_only'] else ('Excluded_not_interventional' if st and st!='INTERVENTIONAL' else 'Pending'))
  base=t['JSON_Record_Path']+'.protocolSection'
  for kind in ['primaryOutcomes','secondaryOutcomes','otherOutcomes']:
   for j,x in enumerate(ps.get('outcomesModule',{}).get(kind,[])):
    rid=f'{n}:{kind}:{j}';rawout[rid]=x
    assert rid in byid
    assert all((x.get(k,'') or '')==(byid[rid][kk] or '') for k,kk in [('measure','measure'),('description','description'),('timeFrame','timeframe')])
  audit.append({'NCT ID':n,'Disease':'; '.join(d for d in ['ASD','ADHD','Tourette','Fragile X'] if d in str(t['Supported_Group'])),'Minimum age raw':lo,'Maximum age raw':hi,'Minimum years standardized':float(age_days(lo)/Fraction(36525,100)) if age_days(lo) is not None else None,'Maximum years standardized':float(age_days(hi)/Fraction(36525,100)) if age_days(hi) is not None else None,'Age status':ast,'Age reason':reason,'Study type':st,'DRUG records':len(drugs),'Drug evidence status':ds,'Active drug names':'; '.join(str(x['Original_Intervention_Name']) for x in active),'Historical drug roles':'; '.join(sorted(set(x['Drug_Role'] for x in roles[n]))),'Eligibility status':status,'Outcomes':sum(r['nct']==n for r in R),'Source JSON':str(p),'Source SHA256':manifest[str(p)],'Eligibility path':base+'.eligibilityModule','Intervention path':base+'.armsInterventionsModule.interventions','Drug role source':str(stage6)+'!Intervention_Final; filter NCT_ID='+n,'Historical age category (not used)':t['Age_Category'],'Title':ps['identificationModule'].get('briefTitle','')})
 assert len(rawout)==4653 and not ageconf
 elig={x['NCT ID'] for x in audit if x['Eligibility status']=='Eligible_structured'};E=[r for r in R if r['nct'] in elig and r['usable_text']]
 assert len({r['outcome_id'] for r in E})==len(E)
 old=json.loads((P/'03_annotation/workbook_payload.json').read_text(encoding='utf8'))['ANNOTATION_WORKSHEET']['Annotation'];seen_ncts={r['NCT ID'] for r in old}
 seen_groups={r['strict_split_group'] for r in R if r['nct'] in seen_ncts}
 # Lock test based only on disease memberships and NCTs, never predictions or human labels.
 rng=random.Random(20261123);groups=defaultdict(list)
 for r in E:groups[r['strict_split_group']].append(r)
 fresh=[r for r in E if r['strict_split_group'] not in seen_groups]
 def select_nct_sample(candidates,k):
  per=defaultdict(list)
  for r in candidates:per[r['nct']].append(r)
  ncts=sorted(per);rng.shuffle(ncts);selected=[]
  for n in ncts[:k]:
   arr=sorted(per[n],key=lambda r:r['outcome_id']);selected.append(arr[rng.randrange(len(arr))])
  return selected
 test=select_nct_sample(fresh,60)
 assert len(test)==60,f'Only {len(test)} fresh independent NCTs; do not fabricate 60/40 feasibility'
 test_groups={r['strict_split_group'] for r in test}
 train=[r for r in E if r['strict_split_group'] not in test_groups]
 pool=select_nct_sample(train,40);assert len(pool)==40
 pool_groups={r['strict_split_group'] for r in pool}
 pilot=select_nct_sample([r for r in train if r['strict_split_group'] not in pool_groups],100)
 if len(pilot)<100:
  # pilot allowed within AL-pool NCTs, but never the human sample objects nor heldout groups.
  ids={r['outcome_id'] for r in pilot+pool};extra=[r for r in train if r['outcome_id'] not in ids];rng.shuffle(extra);pilot+=extra[:100-len(pilot)]
 assert len(pilot)==100 and not {r['outcome_id'] for r in pilot}&{r['outcome_id'] for r in pool}
 labels={'ClinicalSymptoms':['SocialCommunication','IrritabilityAggression','RepetitiveBehavior','AttentionHyperactivity','AnxietyEmotion','Sleep','TicSeverity'],'FunctionAndCognition':['AdaptiveFunction','GlobalFunction','CognitionLanguage'],'BiologicalAndExposure':['BiologicalMechanistic','Pharmacokinetics','GrowthMetabolic'],'Safety':['SafetyTolerability'],'OtherDomain':['Other']}
 mapping={x:k for k,v in labels.items() for x in v}
 rules=json.loads((P/'03_annotation/label_rules.json').read_text())
 instruments={'SocialCommunication':r'\bSRS(?:-2)?\b|Social Responsiveness Scale','IrritabilityAggression':r'Aberrant Behavior Checklist.{0,25}Irritability|\bABC[- ]I\b','RepetitiveBehavior':r'Repetitive Behavior Scale|\bRBS[- ]R\b','AttentionHyperactivity':r'ADHD Rating Scale|\bADHD[- ]RS\b|Swanson.{0,25}Nolan|\bSNAP[- ]IV\b','Sleep':r'Children.s Sleep Habits|Pittsburgh Sleep Quality','AdaptiveFunction':r'Vineland|Adaptive Behavior Assessment','GlobalFunction':r'Clinical Global Impression|\bCGI[- ](?:I|S)\b','TicSeverity':r'Yale Global Tic Severity|\bYGTSS\b','CognitionLanguage':r'Wechsler|Peabody Picture Vocabulary|Mullen Scales'}
 def rule_record(r):
  text=r['measure']+'\n'+r['description'];hits=[];fine=set();families={}
  for family,config in [('keywords',rules),('instrument_dictionary',instruments)]:
   labs=set()
   for lab,pattern in config.items():
    for m in re.finditer(pattern,text,re.I):hits.append({'source':family,'leaf':lab,'coarse':mapping[lab],'evidence':m.group(0)});fine.add(lab);labs.add(mapping[lab])
   families[family]=sorted(labs)
  candidates=sorted({mapping[x] for x in fine});conflict=bool(families['keywords'] and families['instrument_dictionary'] and families['keywords']!=families['instrument_dictionary'])
  return {'record_id':r['outcome_id'],'rule_leaf_candidates':sorted(fine),'rule_labels':candidates,'rule_status':'Abstain' if not candidates else ('Rule_sources_disagree' if conflict else 'Rule_candidate'),'family_votes':families,'evidence':hits,'llm_labels':None,'llm_status':'Not executed: no verified local LLM','aggregate_labels':None,'aggregate_status':'Pending_LLM','uncertainty':None,'uncertainty_reason':'Model/LLM unavailable; rule coverage is not calibrated probability'}
 weak=[rule_record(r) for r in train];wby={x['record_id']:x for x in weak};pilot_rows=[]
 for r in pilot:
  x=wby[r['outcome_id']]
  pilot_rows.append({'Record ID':r['outcome_id'],'NCT ID':r['nct'],'Disease':'; '.join(r['diseases']),'Outcome Type':r['outcome_type'],'Outcome Measure':r['measure'],'Outcome Description':r['description'],'Time Frame':r['timeframe'],'Rule candidates':'; '.join(x['rule_labels']),'Rule leaf candidates':'; '.join(x['rule_leaf_candidates']),'Rule evidence':json.dumps(x['evidence'],ensure_ascii=False),'Rule status':x['rule_status'],'LLM status':x['llm_status'],'LLM model':None,'LLM version':None,'Prompt version':'ndd-coarse-multilabel-v1.0','LLM labels':None,'LLM evidence':None,'LLM error':'No accessible local model has been verified; inference not run','Aggregate labels':None,'Aggregate status':'Pending_LLM','Uncertainty score':None,'Source JSON':r['source_file'],'JSON path':r['source_outcome_path'],'Source SHA256':r['source_hash']})
 human=[]
 for partition,rows in [('Locked_Test',test),('AL_Reference_Pool',pool)]:
  for r in rows:
   human.append({'Record ID':r['outcome_id'],'NCT ID':r['nct'],'Partition':partition,'Disease':'; '.join(r['diseases']),'Outcome Type':r['outcome_type'],'Outcome Measure':r['measure'],'Outcome Description':r['description'],'Time Frame':r['timeframe'],'Annotator 1 Label':None,'Annotator 2 Label':None,'Adjudicated Label':None,'Evidence / Reason':None,'Review Status':None,'Annotator 1 Name':None,'Annotator 2 Name':None,'Adjudicator Name':None,'Verified Instrument':None,'Strict Split Group':r['strict_split_group'],'Source JSON':r['source_file'],'JSON Path':r['source_outcome_path'],'Source SHA256':r['source_hash']})
 with (P/'02_processed_data/near_duplicate_pairs.csv').open(encoding='utf-8-sig') as f:pairs=list(csv.DictReader(f))
 testn={r['nct'] for r in test};pooln={r['nct'] for r in pool};trainn={r['nct'] for r in train};testids={r['outcome_id'] for r in E if r['strict_split_group'] in test_groups};trainids={r['outcome_id'] for r in train}
 nearcross=[x for x in pairs if (x['outcome_a'] in testids and x['outcome_b'] in trainids) or (x['outcome_b'] in testids and x['outcome_a'] in trainids)]
 assert not testn&pooln and not trainn&testn and not nearcross
 assert not {r['normalized_hash'] for r in E if r['outcome_id'] in testids}&{r['normalized_hash'] for r in train}
 profile=lambda rows:{'records':len(rows),'nct':len({r['nct'] for r in rows}),'disease_memberships':dict(Counter(d for r in rows for d in r['diseases'])),'types':dict(Counter(r['outcome_type'] for r in rows))}
 metrics={'version':'integrated-pediatric-ws-al-v1.0.0','timestamp':stamp,'source_trials':670,'source_outcomes':4653,'trial_status':dict(Counter(x['Eligibility status'] for x in audit)),'age_status':dict(Counter(x['Age status'] for x in audit)),'drug_status':dict(Counter(x['Drug evidence status'] for x in audit)),'eligible_trials':len(elig),'eligible_profile':profile(E),'test_profile':profile(test),'pool_profile':profile(pool),'pilot_profile':profile(pilot),'training_profile':profile(train),'fresh_test_candidates':profile(fresh),'quarantined_test_groups':len(test_groups),'quarantined_test_outcomes':len(testids),'quarantined_test_ncts':len({r['nct'] for r in E if r['outcome_id'] in testids}),'test_vs_train_near_edges':len(nearcross),'test_vs_pool_nct_overlap':len(testn&pooln),'human_labels':0,'LLM_executed_records':0,'rule_labeled_training_records':sum(bool(x['rule_labels']) for x in weak),'rule_abstentions':sum(not x['rule_labels'] for x in weak),'rule_family_disagreements':sum(x['rule_status']=='Rule_sources_disagree' for x in weak),'age_field_conflicts':ageconf,'known_previous_annotation_ncts':len(seen_ncts),'rule_candidate_coarse_distribution':dict(Counter(l for x in weak for l in x['rule_labels']))}
 dump('audit/metrics.json',metrics);dump('audit/input_manifest.json',manifest);dump('data/trial_eligibility_audit.json',audit);dump('data/pediatric_outcomes.json',E);dump('data/training_weak_labels.json',weak)
 dump('data/locked_partitions.json',{'seed':20261123,'status':'IDs frozen before LLM pilot; taxonomy and human labels pending approval','test_ids':[r['outcome_id'] for r in test],'pool_ids':[r['outcome_id'] for r in pool],'pilot_ids':[r['outcome_id'] for r in pilot],'test_groups':sorted(test_groups),'training_ids':[r['outcome_id'] for r in train]})
 dump('data/llm_pilot_input.json',[{k:r[k] for k in ['outcome_id','measure','description','timeframe']} for r in pilot]);dump('data/coarse_taxonomy.json',labels)
 dump('audit/workbook_payload.json',{'WEAK_SUPERVISION_PILOT':{'Pilot_100':pilot_rows,'Rule_Training_Cohort':[{'Record ID':x['record_id'],'Rule labels':'; '.join(x['rule_labels']),'Status':x['rule_status'],'LLM status':x['llm_status'],'Aggregate status':x['aggregate_status'],'Evidence':json.dumps(x['evidence'],ensure_ascii=False)} for x in weak],'Trial_Eligibility_670':audit,'Taxonomy':[{'Coarse label':k,'Leaf labels':'; '.join(v),'Approval':'Awaiting clinical confirmation'} for k,v in labels.items()]},'HUMAN_ANNOTATION_100':{'Blinded_Annotation':human,'Instructions':[{'Instruction':'Label only measured constructs. Candidate/model predictions are deliberately absent. Use the coarse labels on Taxonomy; separate multiple labels with semicolons. Unclear is unresolved, not a negative.'},{'Instruction':'Two independent reviewers work in separate copies. Budget is 100 distinct records (60 test, 40 pool); two reviewers imply at least 200 labeling actions plus adjudication.'},{'Instruction':'Locked_Test labels must never tune prompts, taxonomy, thresholds, models or queries. Pool labels are revealed only after a recorded query. Keep all human fields blank until genuine review.'}],'Taxonomy':[{'Coarse label':k,'Leaf labels':'; '.join(v),'Approval':'Awaiting clinical confirmation'} for k,v in labels.items()]}})
 print(json.dumps(metrics,ensure_ascii=False))
if __name__=='__main__':main()

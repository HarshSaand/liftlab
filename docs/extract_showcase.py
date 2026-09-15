from pathlib import Path
import argparse, json, hashlib, sys
p=argparse.ArgumentParser();p.add_argument('--source', type=Path, required=True);a=p.parse_args()
S=a.source.resolve(); D=Path(__file__).resolve().parent; R=D.parent
D.mkdir(exist_ok=True)
import numpy as np
import pandas as pd
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for c in iter(lambda:f.read(1048576),b''):h.update(c)
 return h.hexdigest()
def source(rel):return {'path':rel,'sha256':digest(S/rel)}
def write(data):
 import subprocess
 data['dataset_url']='https://ailab.criteo.com/criteo-uplift-prediction-dataset/'
 data['source_code_commit']=subprocess.check_output(['git','-C',str(R),'rev-parse','HEAD'],text=True).strip()
 data['extractor_sha256']=digest(Path(__file__))
 data['sources']=sources
 data['source_repository']='https://github.com/HarshSaand/'+R.name
 data['extraction']='python docs/extract_showcase.py --source /path/to/reproduced/project'
 (D/'output-example.json').write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n')
import joblib
sys.path.insert(0,str(R));from liftlab import s_features
f='data/sample.parquet';df=pd.read_parquet(S/f);x=df[df['_split']=='test'].head(8);model=joblib.load(S/'checkpoints/models.joblib');features=model['features'];v=x[features].to_numpy(dtype=np.float32)
s=model['s'].predict_proba(s_features(v,1))[:,1]-model['s'].predict_proba(s_features(v,0))[:,1];tt=model['t1'].predict_proba(v)[:,1]-model['t0'].predict_proba(v)[:,1];dr=model['dr'].predict(v)
raw=[dict(sample_row=int(idx),covariate_group=str(r['_group']),logistic_s_effect=float(s[i]),boosted_t_effect=float(tt[i]),crossfit_dr_effect=float(dr[i])) for i,(idx,r) in enumerate(x.iterrows())]
sources=[source(f),source('checkpoints/models.joblib'),source('outputs/provenance.json')]
rows=[[str(r['sample_row']),f"{r['logistic_s_effect']:+.5f}",f"{r['boosted_t_effect']:+.5f}",f"{r['crossfit_dr_effect']:+.5f}"] for r in raw]
write(dict(title='LiftLab',subtitle='Real held-out records become treatment-effect scores',eyebrow='ACTUAL CHECKPOINT INFERENCE',context='Criteo randomized advertising study · first 8 held-out sample rows',columns=['Sample row','Logistic S effect','Boosted T effect','Cross-fit DR effect'],rows=rows,raw=raw,note='Effects concern visit probability, not conversion or revenue. These are model estimates: individual counterfactual effects cannot be verified. Raw anonymized feature rows stay outside Git; row IDs and source hashes support reproduction.',input='12 pre-treatment anonymous covariates',output='Three per-record treatment-effect estimates'))

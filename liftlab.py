"""Seeded real-data incrementality benchmark; no synthetic benchmark fallback."""
import os
for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '2'
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, LGBMRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
URL = 'https://huggingface.co/datasets/criteo/criteo-uplift/resolve/2424920/criteo-research-uplift-v2.1.csv.gz'
EXPECTED_ROWS = 13979592
SEED = 42

def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()

def fetch():
    DATA.mkdir(exist_ok=True)
    path = DATA / 'criteo-research-uplift-v2.1.csv.gz'
    if not path.exists():
        temporary = path.with_suffix('.partial')
        with requests.get(URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with temporary.open('wb') as f:
                for b in r.iter_content(1048576):
                    f.write(b)
        temporary.rename(path)
    return path

def covariate_hash(frame):
    return pd.util.hash_pandas_object(frame, index=False).to_numpy(dtype=np.uint64)

def split_groups(hashes):
    """Same covariates always get same partition, independent of row order."""
    buckets = hashes % 100
    return np.where(buckets < 70, 'train', np.where(buckets < 85, 'dev', 'test'))

def prepare(n):
    """Uniform priority sample across the WHOLE corrected file, not a prefix."""
    path = fetch()
    rng = np.random.default_rng(SEED)
    sample = None
    count = 0
    for chunk in pd.read_csv(path, chunksize=250000):
        chunk['_priority'] = rng.random(len(chunk))
        count += len(chunk)
        sample = chunk if sample is None else pd.concat([sample, chunk], ignore_index=True)
        if len(sample) > n:
            chosen = np.argpartition(sample['_priority'].to_numpy(), n - 1)[:n]
            sample = sample.iloc[chosen].copy()
        if count % 2000000 == 0:
            print(f'sampled through {count:,} real records', flush=True)
    if count != EXPECTED_ROWS:
        raise ValueError(f'Expected corrected v2.1 {EXPECTED_ROWS} rows, found {count}')
    sample = sample.sort_values('_priority').drop(columns='_priority').reset_index(drop=True)
    features = [c for c in sample if c.startswith('f') and c[1:].isdigit()]
    if len(features) != 12 or not np.isfinite(sample[features]).all().all():
        raise ValueError('Unexpected or nonfinite covariates')
    # Hash original float64 covariates BEFORE float32 model conversion.
    sample['_group'] = covariate_hash(sample[features])
    sample['_split'] = split_groups(sample['_group'].to_numpy())
    sample.to_parquet(DATA / 'sample.parquet', index=False)
    OUT.mkdir(exist_ok=True)
    provenance = dict(url=URL, dataset='Criteo uplift corrected v2.1', source_rows=count,
        source_sha256=sha256(path), sample_rows=len(sample), seed=SEED,
        method='Uniform random-priority sampling over entire source, float64 covariate-group hash partitions',
        sample_sha256=sha256(DATA / 'sample.parquet'),
        splits=sample['_split'].value_counts().to_dict(),
        exact_duplicate_covariate_rows=int(sample['_group'].duplicated().sum()),
        features=features, outcome='visit', excluded=['conversion', 'exposure', 'treatment as covariate'],
        source_license='CC BY-NC-SA 4.0; raw data not redistributed')
    (OUT / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return sample

def learner():
    return LGBMClassifier(n_estimators=120, learning_rate=.05, num_leaves=15,
        min_child_samples=200, reg_lambda=5, n_jobs=2, verbosity=-1, random_state=SEED)

def s_features(x, t):
    t = np.broadcast_to(np.asarray(t, dtype=np.float32), (len(x),)).reshape(-1, 1)
    return np.concatenate([x, t, x*t], axis=1)

def transformed_outcome(y, t, p):
    if not 0 < p < 1:
        raise ValueError('Propensity must lie strictly between zero and one')
    return y * (t/p - (1-t)/(1-p))

def curve(score, pseudo):
    order = np.argsort(-score, kind='stable')
    gain = np.r_[0., np.cumsum(pseudo[order]) / len(pseudo)]
    fractions = np.linspace(0, 1, len(gain))
    auuc = float(np.trapezoid(gain, fractions))
    return fractions, gain, auuc, auuc - float(gain[-1])/2

def metrics(score, y, t, p, bootstrap=200, score_is_effect=True):
    pseudo = transformed_outcome(y, t, p)
    fractions, gain, auuc, qini = curve(score, pseudo)
    order = np.argsort(-score, kind='stable')
    budgets = {}
    # Pairs bootstrap of frozen test policy masks, conditional on fitted model/p.
    rng = np.random.default_rng(SEED)
    for budget in [.1, .2, .3]:
        mask = np.zeros(len(y)); mask[order[:round(len(y)*budget)]] = 1
        contribution = mask*pseudo*1000
        boot = [float(contribution[rng.integers(0, len(y), len(y))].mean()) for _ in range(bootstrap)]
        budgets[str(budget)] = dict(incremental_visits_per_1000_population=float(contribution.mean()),
            bootstrap_95_ci=np.quantile(boot, [.025, .975]).tolist(),
            targeted=int(mask.sum()))
    bins = []
    for indexes in np.array_split(order, 10):
        bins.append(dict(n=len(indexes), mean_prediction=float(score[indexes].mean()),
            observed_ipw_effect=float(pseudo[indexes].mean()),
            treated=int(t[indexes].sum()), visits=int(y[indexes].sum())))
    if not score_is_effect:
        for item in bins:
            item['mean_response_probability']=item.pop('mean_prediction')
    return dict(auuc=auuc, qini_area=qini, budgets=budgets,
        **{('effect_bins' if score_is_effect else 'response_score_bins'):bins})

def run():
    started = time.time()
    OUT.mkdir(exist_ok=True)
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    df = pd.read_parquet(DATA/'sample.parquet')
    features = [c for c in df if c.startswith('f') and c[1:].isdigit()]
    x = df[features].to_numpy(dtype=np.float32)
    y = df.visit.to_numpy(dtype=np.float32)
    t = df.treatment.to_numpy(dtype=np.float32)
    train = df['_split'].to_numpy() == 'train'
    dev = df['_split'].to_numpy() == 'dev'
    test = df['_split'].to_numpy() == 'test'
    p = float(t[train].mean())
    groups = df['_group'].to_numpy()
    assert set(groups[train]).isdisjoint(groups[test])
    assert set(groups[train]).isdisjoint(groups[dev])
    assert set(groups[dev]).isdisjoint(groups[test])
    print('Training logistic S-learner', flush=True)
    s = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, C=1., random_state=SEED))
    s.fit(s_features(x[train], t[train]), y[train])
    print('Training T-learner', flush=True)
    m0, m1 = learner(), learner()
    m0.fit(x[train & (t==0)], y[train & (t==0)])
    m1.fit(x[train & (t==1)], y[train & (t==1)])
    print('Training group-cross-fitted doubly robust learner', flush=True)
    xi, yi, ti = x[train], y[train], t[train]
    # An independent hash bit provides within-training group-safe two-fold OOF predictions.
    folds = (groups[train] // 100) % 2
    pseudo = np.zeros(len(xi), dtype=np.float64)
    for fold in [0, 1]:
        fit, hold = folds != fold, folds == fold
        a, b = learner(), learner()
        a.fit(xi[fit & (ti==0)], yi[fit & (ti==0)])
        b.fit(xi[fit & (ti==1)], yi[fit & (ti==1)])
        mu0, mu1 = a.predict_proba(xi[hold])[:,1], b.predict_proba(xi[hold])[:,1]
        pf = float(ti[fit].mean())
        pseudo[hold] = mu1-mu0 + ti[hold]*(yi[hold]-mu1)/pf - (1-ti[hold])*(yi[hold]-mu0)/(1-pf)
    dr = LGBMRegressor(n_estimators=120, learning_rate=.05, num_leaves=15,
        min_child_samples=400, reg_lambda=10, random_state=SEED, n_jobs=2, verbosity=-1)
    dr.fit(xi, pseudo)
    joblib.dump(dict(s=s, t0=m0, t1=m1, dr=dr, features=features, propensity=p), ROOT/'checkpoints/models.joblib')
    report = dict(outcome='visit (not conversion or revenue)', sample_rows=len(df),
        split_counts={part:int((df['_split']==part).sum()) for part in ['train','dev','test']},
        train_treatment_probability=p, seed=SEED, bootstrap_repetitions=200,
        protocol='Fixed hyperparameters, group-disjoint partitions; no test model/threshold selection',
        runtime=dict(python=platform.python_version(), platform=platform.platform(), threads=2),
        splits={}, limitations=['One million sampled rows, not full-dataset training',
            'Randomized-trial benchmark altered by anonymization/subsampling; no business revenue claim',
            'No time or advertiser identifiers; no temporal or advertiser generalization assessment',
            'Bootstrap conditional on trained model and estimated training propensity; no training-seed uncertainty',
            'Unobserved individual counterfactuals prohibit individual-effect accuracy measurement'])
    fig, ax = plt.subplots(figsize=(8,5))
    for name, mask in [('dev', dev), ('test', test)]:
        xx = x[mask]
        scores = {'logistic_s': s.predict_proba(s_features(xx,1))[:,1]-s.predict_proba(s_features(xx,0))[:,1],
            'boosted_t': m1.predict_proba(xx)[:,1]-m0.predict_proba(xx)[:,1],
            'crossfit_dr': dr.predict(xx), 'response_targeting': m1.predict_proba(xx)[:,1]}
        values = {key:metrics(value, y[mask], t[mask], p,
            score_is_effect=(key!='response_targeting')) for key,value in scores.items()}
        ate = float(transformed_outcome(y[mask], t[mask], p).mean())
        values['random_policy_expectation'] = dict(average_effect=ate,
            budgets={str(b):1000*b*ate for b in [.1,.2,.3]})
        report['splits'][name] = dict(n=int(mask.sum()), visits=int(y[mask].sum()),
            treated=int(t[mask].sum()), models=values)
        if name == 'test':
            for key,value in scores.items():
                f,g,_,_ = curve(value, transformed_outcome(y[mask], t[mask],p))
                ax.plot(f[::max(1,len(f)//1000)],1000*g[::max(1,len(f)//1000)],label=key)
            ax.plot([0,1],[0,1000*ate],'k--',label='random expectation')
    report['elapsed_seconds'] = time.time()-started
    report['checkpoint_sha256'] = sha256(ROOT/'checkpoints/models.joblib')
    (OUT/'evaluation.json').write_text(json.dumps(report, indent=2))
    ax.set(xlabel='Fraction of held-out population targeted',ylabel='IPW incremental visits per 1,000 population',
        title='Frozen-policy held-out uplift curves (real Criteo sample)')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/'uplift-curves.png',dpi=160)
    print(json.dumps({'splits':report['split_counts'],'elapsed_seconds':report['elapsed_seconds'],
        'test_qini':{k:v.get('qini_area') for k,v in report['splits']['test']['models'].items()}},indent=2),flush=True)

def predict(input_path, output_path):
    """Score pre-treatment covariates with a locally trained, trusted checkpoint."""
    model=joblib.load(ROOT/'checkpoints/models.joblib')
    frame=pd.read_csv(input_path)
    x=frame[model['features']].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all(): raise ValueError('Inputs must be finite')
    result=pd.DataFrame({
        'logistic_s_effect':model['s'].predict_proba(s_features(x,1))[:,1]-model['s'].predict_proba(s_features(x,0))[:,1],
        'boosted_t_effect':model['t1'].predict_proba(x)[:,1]-model['t0'].predict_proba(x)[:,1],
        'crossfit_dr_effect':model['dr'].predict(x)})
    result.to_csv(output_path,index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare','run','predict'])
    parser.add_argument('--rows', type=int, default=1000000)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path,default=DATA/'scores.csv')
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.rows < 1000: parser.error('Use at least 1,000 real rows')
        prepare(args.rows)
    elif args.command=='run': run()
    else:
        if args.input is None: parser.error('predict requires --input')
        predict(args.input,args.output)

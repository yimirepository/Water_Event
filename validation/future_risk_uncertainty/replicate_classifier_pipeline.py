"""Reproduce the supplied RF classifier and bootstrap its 2050 risk probabilities.

Also runs the five chronological historical folds used for reviewer validation.
The binary training/validation outcome follows predict_conflict.ipynb: conflict OR both.
The historical validation feature uses conflict counts through t-1, to ensure that
an outcome is not used to construct its own risk score.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap_77_suitability as fit

ROOT=Path(__file__).resolve().parents[1]
PANEL_PATH=fit.HIST_FILE
FUTURE_PATH=fit.FUTURE_FILE
FEATURES=['Average_conflict','population_density','export','gdp_percapita','rural_population','precipitation','bws_score']
FOLDS=[(1984,1991,1995),(1990,1997,2001),(1996,2003,2007),(2002,2009,2013),(2008,2015,2019)]
REPS=500
SEED=20261109

def panel_prepare():
 d=pd.read_csv(PANEL_PATH).replace('..',np.nan).sort_values(['country','year']).copy()
 d['conflict']=pd.to_numeric(d['conflict'],errors='coerce').fillna(0)
 d['both']=pd.to_numeric(d['both'],errors='coerce').fillna(0)
 d['conflict_target']=((d.conflict.eq(1))|(d.both.eq(1))).astype(int)
 d['Average_conflict']=d.groupby('country').conflict_count_yr.transform(lambda x:x.cumsum()/np.arange(1,len(x)+1))
 d['past_average_conflict']=d.groupby('country').conflict_count_yr.transform(lambda x:x.shift(1).expanding(min_periods=1).mean())
 return d

def future_prepare(panel):
 f=pd.read_csv(FUTURE_PATH).dropna(subset=['conflict_probability','gdp_constant_2015_us','export_2004_percent','bws_score','precipitation']).copy().reset_index(drop=True)
 f['gdp_percapita']=f['gdp_constant_2015_us'];f['export']=f['export_2004_percent'];f['rural_population']=f['rural_population_percent']
 latest=panel.loc[panel.year.eq(2019)].drop_duplicates('iso3').set_index('iso3')['Average_conflict']
 f['Average_conflict']=f.iso3.map(latest)
 return f

def scaler_pair(train, fut, cols):
 # Matches the notebook's StandardScaler over the joined historical and future inputs.
 allx=pd.concat([train[cols],fut[cols]],ignore_index=True)
 scaler=StandardScaler().fit(allx)
 return scaler.transform(train[cols]),scaler.transform(fut[cols])

def model_factory(seed=42): return RandomForestClassifier(random_state=seed,n_jobs=1)

def make_cv(panel):
 rows=[]; scored=[]
 for j,(end,start,stop) in enumerate(FOLDS,1):
  usable=panel.dropna(subset=['conflict_target','past_average_conflict','population_density','export','gdp_percapita','rural_population','precipitation','bws_score']).copy()
  tr=usable[usable.year.le(end)].copy(); te=usable[usable.year.between(start,stop)].copy()
  cols=['past_average_conflict','population_density','export','gdp_percapita','rural_population','precipitation','bws_score']
  xtr,xte=scaler_pair(tr,te,cols)
  m=model_factory(42);m.fit(xtr,tr.conflict_target.astype(int))
  p=m.predict_proba(xte)[:,1];y=te.conflict_target.to_numpy(int)
  rows.append({'fold':j,'train_through':end,'test_period':f'{start}-{stop}','n':len(te),'events':int(y.sum()),'event_rate':y.mean(),'auc':roc_auc_score(y,p),'brier':brier_score_loss(y,p),'average_precision':average_precision_score(y,p)})
  scored.append(pd.DataFrame({'iso3':te.iso3,'country':te.country,'year':te.year,'observed':y,'risk_score':p,'fold':j}))
 out=pd.concat(scored,ignore_index=True);n_top=int(np.ceil(len(out)*.10));top=out.nlargest(n_top,'risk_score')
 summ={'heldout_n':len(out),'heldout_events':int(out.observed.sum()),'countries':out.country.nunique(),'event_rate':out.observed.mean(),'pooled_auc':roc_auc_score(out.observed,out.risk_score),'pooled_brier':brier_score_loss(out.observed,out.risk_score),'pooled_average_precision':average_precision_score(out.observed,out.risk_score),'top_decile_n':len(top),'top_decile_events':int(top.observed.sum()),'top_decile_event_rate':top.observed.mean()}
 return pd.DataFrame(rows),pd.DataFrame([summ]),out

def cluster_bootstrap_panel(panel,rng):
 ids=panel.country.dropna().drop_duplicates().to_numpy();draws=rng.choice(ids,len(ids),replace=True);parts=[]
 by={k:v for k,v in panel.groupby('country',sort=False)}
 for b,c in enumerate(draws):
  q=by[c].copy();q['_boot_country']=b;parts.append(q)
 return pd.concat(parts,ignore_index=True)

def train_projector(train,future,seed=42):
 cols=['Average_conflict','population_density','export','gdp_percapita','rural_population','precipitation','bws_score']
 train=train.dropna(subset=['conflict_target']+cols).copy()
 future=future.dropna(subset=cols).copy()
 xt,xf=scaler_pair(train,future,cols)
 m=model_factory(seed);m.fit(xt,train.conflict_target.astype(int))
 return m.predict_proba(xf)[:,1],future.iso3.to_numpy()

def main():
 panel=panel_prepare();future=future_prepare(panel)
 # Historical CV; predictions tested against observed binary Conflict/Both outcome.
 fold,cvsummary,cvscore=make_cv(panel)
 # Full available-panel classifier vs probabilities supplied to the score engine.
 p_full,ids=train_projector(panel,future,42)
 sup=future.conflict_probability.to_numpy();assert np.array_equal(ids,future.iso3.to_numpy())
 corr=np.corrcoef(p_full,sup)[0,1];mae=np.mean(np.abs(p_full-sup))
 scores=fit.score_mode_metrics(fit.PUBLISHED_RAW);benchmark=fit.historical_fit_scores(panel,scores)
 point_all,n_all,long=fit.any_good_share(future,scores,benchmark)
 point_country=long.groupby('iso3').good.any()
 high=future.cluster_risk.eq(1)
 hi_ids=future.loc[high,'iso3'];hi_n=int(point_country.reindex(hi_ids).fillna(False).sum());point_hi=hi_n/int(high.sum())
 rng=np.random.default_rng(SEED);records=[]
 for b in range(REPS):
  bp=cluster_bootstrap_panel(panel,rng)
  p_b,ids_b=train_projector(bp,future,42+b+1)
  # Recenter model variation on the supplied probabilities so bootstrap draws
  # reflect classifier uncertainty without replacing the reported point inputs.
  p_adj=np.clip(sup+(p_b-p_full),0,1)
  _,_,long_b=fit.any_good_share(future,scores,benchmark,p_adj)
  flags=long_b.groupby('iso3').good.any()
  records.append({'replicate':b+1,'all_country_share':float(flags.mean()),'high_risk_share':float(flags.reindex(hi_ids).fillna(False).mean())})
  if (b+1)%100==0:print(f'Completed {b+1}/{REPS}',flush=True)
 out=ROOT/'r2c77_uncertainty'/'notebook_aligned_500';out.mkdir(parents=True,exist_ok=True)
 fold.to_csv(out/'historical_cv_folds.csv',index=False);cvsummary.to_csv(out/'historical_cv_summary.csv',index=False);cvscore.to_csv(out/'historical_cv_predictions.csv',index=False)
 reps=pd.DataFrame(records);reps.to_csv(out/'classifier_bootstrap_replicates.csv',index=False)
 summ=[]
 for key,point,num,den in [('all_country_share',point_all,n_all,len(future)),('high_risk_share',point_hi,hi_n,int(high.sum()))]:
  q=reps[key].quantile([.025,.5,.975]);summ.append({'group':key,'numerator':num,'denominator':den,'point_share':point,'bootstrap_mean':reps[key].mean(),'ci_2_5':q.loc[.025],'bootstrap_median':q.loc[.5],'ci_97_5':q.loc[.975],'sd':reps[key].std(ddof=1)})
 pd.DataFrame(summ).to_csv(out/'classifier_bootstrap_summary.csv',index=False)
 pd.DataFrame([{'classifier_refit_vs_supplied_correlation':corr,'classifier_refit_vs_supplied_mae':mae,'historical_score_benchmark_n':len(benchmark),'historical_mode_adoption_rows':int(panel[fit.MODES].fillna(0).gt(0).sum().sum())}]).to_csv(out/'point_reproduction_diagnostics.csv',index=False)
 print('\nHistorical CV summary\n',cvsummary.to_string(index=False));print('\nProbability comparison correlation',corr,'MAE',mae)
 print('\nBootstrap summary\n',pd.DataFrame(summ).to_string(index=False))

if __name__=='__main__':main()

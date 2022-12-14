import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
datadir = r"/content/gdrive/MyDrive/bosch_data"  
featuresdir = r"/content/gdrive/MyDrive/bosch_feature_engineering" 
resultsdir = r"/content/gdrive/MyDrive/bosch_results"

train_date = get_cols_used('train_date')
hash_path_flow1 = hash_path_flow(train_date)
time_all_feats1 = time_all_feats(train_date)
line_time_feats1 = line_time_feats(train_date)
station_time_feats1 = station_time_feats(train_date)

X_train_numeric = get_cols_used('train_numeric')
y_train_numeric = get_response().values
train_numeric_feats = get_numeric_feats(X_train_numeric,y_train_numeric,'train_numeric')

train_all_features = pd.concat([hash_path_flow1, time_all_feats1, line_time_feats1, station_time_feats1,train_numeric_feats],axis=1)
save_pickle(train_all_features,os.path.join(featuresdir,'train_all_feats.pickle'))
del train_date, hash_path_flow1, time_all_feats1, line_time_feats1, station_time_feats1, train_numeric_feats, X_train_numeric, y_train_numeric
gc.collect()

test_date = get_cols_used('test_date')
hash_path_flow2 = hash_path_flow(test_date)
time_all_feats2 = time_all_feats(test_date)
line_time_feats2 = line_time_feats(test_date)
station_time_feats2 = station_time_feats(test_date)

X_test_numeric = get_cols_used('test_numeric')
y_test_numeric = get_response().values
test_numeric_feats = get_numeric_feats(X_test_numeric,y_test_numeric,'test_numeric')

test_all_features = pd.concat([hash_path_flow2, time_all_feats2, line_time_feats2, station_time_feats2, test_numeric_feats],axis=1)
save_pickle(test_all_features,os.path.join(featuresdir,'test_all_feats.pickle'))
del test_date, hash_path_flow2, time_all_feats2, line_time_feats2, station_time_feats2, test_numeric_feats, X_test_numeric, y_test_numeric
gc.collect()

# functions
import pickle
def save_pickle(x, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
def read_pickle(filename):
    with open(filename, 'rb') as handle:
        x = pickle.load(handle)
    return x

def get_cols_used(file,nrows=None):
  cols = pd.read_csv(os.path.join(datadir,file+'.csv.zip'),nrows=0,compression='zip').columns
  cols_used = np.setdiff1d(cols,'Response')
  types_dict = {'Id': np.int32}
  types_dict.update({col: np.float16 for col in cols_used if col not in types_dict})
  df = pd.read_csv(os.path.join(datadir,file+'.csv.zip'),nrows=nrows,index_col=0,compression='zip',usecols=cols_used,dtype=types_dict,engine='c')
  return df

def get_response(nrows=None):
  df = pd.read_csv(os.path.join(datadir,'train_numeric.csv.zip'),nrows=nrows,index_col=0,compression='zip',dtype=np.int8,usecols=['Id','Response'],engine='c')
  return df

def get_ids(file,nrows=None):
  df = pd.read_csv(os.path.join(datadir,file+'.csv.zip'),nrows=nrows,compression='zip',dtype=np.int32,usecols=[0],engine='c')
  return df

def multiIndex_cols(df,prefix):
  dt=df.copy()
  if prefix=='all':
    tuples = [tuple([int(a[1:]) for a in x.split('_')]) for x in df.columns]
    names = ['line','station', 'feature']
  elif prefix=='S':
    tuples = [tuple([int(a[1:]) for a in x[3:].split('_')]) for x in df.columns]
    names = ['station', 'feature']
  else : 
    tuples = [tuple(int(a[1:]) for a in (x.split('_')[0],x.split('_')[2])) for x in df.columns]
    names = ['line', 'feature']
  new_columns = pd.MultiIndex.from_tuples(tuples, names=names)
  dt.columns = new_columns
  return dt

def hash_path_flow(df):
  dt = multiIndex_cols(df,'S')
  new_feats = pd.DataFrame(index=dt.index)
  N = 10000
  n_rows = dt.shape[0]
  new_feats['path_count'] = -1
  new_feats['path_flow'] = -1
  print("path flow start...")
  for i in tqdm(range(n_rows//N+1)):
    start = i*N
    end = min((i+1)*N, n_rows)
    tmp = dt.iloc[start:end]
    new_feats['path_count'].iloc[start:end] = tmp.groupby(level=0, axis=1).min().notnull().sum(axis=1).astype(np.int8)
    new_feats['path_flow'].iloc[start:end] = tmp.groupby(level=0, axis=1).min().apply(
        lambda x: (x>0).astype(int).astype(str), raw=True).apply(
        lambda x: ''.join(x.values), axis=1).apply(lambda x: hash(x) % 2**15).astype(int)
  save_pickle(new_feats, os.path.join(featuresdir,'date_path_feats.pickle'))
  print("path flow success...")
  del dt
  gc.collect()
  return new_feats

def time_extract(dt,prefix):
  # dt: multiIndex_cols
  time_extract = pd.DataFrame(index=dt.index)
  prefix_n = dt.columns.get_level_values(0).unique()
  N = 10000
  n_rows = dt.shape[0]
  for p in prefix_n:
    time_extract[prefix+str(p)+'_start'] = -1
    time_extract[prefix+str(p)+'_end'] = -1
    for i in range(n_rows//N+1):
      start = i*N
      end = min((i+1)*N, n_rows)
      tmp = dt.iloc[start:end]
      time_extract[prefix+str(p)+'_start'].iloc[start:end] = tmp[p].min(axis=1).values
      time_extract[prefix+str(p)+'_end'].iloc[start:end] = tmp[p].max(axis=1).values
    time_extract[prefix+str(p)+'_duration'] = time_extract[prefix+str(p)+'_end'] - time_extract[prefix+str(p)+'_start'] 
  return  time_extract

def station_time_filter(df):
  # df: time_exract
  import re
  duration_cols = re.findall(r'S[0-9]+_duration',' '.join(df.columns))
  temp = pd.Series()
  for f in tqdm(duration_cols):
    bins = int(max(10, df[f].notnull().sum()/20000))
    temp = pd.qcut(df[f], bins, labels=False, duplicates='drop')
    if (len(temp.dropna().unique())==1) | (temp.notnull().sum()<20):
      duration_cols.remove(f)                         
  station_n = re.findall(r'S[0-9]+',' '.join(duration_cols))
  cols_used = [c for c in df.columns if c.split('_')[0] in station_n]
  sation_time_filter = df[cols_used]
  sation_time_filter.to_csv(os.path.join(featuresdir,'filtered_station_time.csv'))
  return sation_time_filter

def id_diff(df, sorts):   
  df.reset_index(inplace=True)
  id_diff = pd.DataFrame(df.Id, index=df.index)
  for c in sorts:
    tem = df[[c,'Id']]
    tem.sort_values([c,'Id'],inplace=True)
    id_diff[c+'_prevIdDiff'] = tem.Id.diff().fillna(9999999)
    id_diff[c+'_nextIdDiff'] = tem.Id.diff(-1).fillna(9999999)
  id_diff.set_index('Id', drop=True, inplace=True)
  return id_diff

def time_transform(df,sorts):
  dt = df.copy()
  dt.reset_index(inplace=True)
  dt.reset_index(inplace=True)
  dt.rename(columns={'index':'idx'}, inplace=True)
  cols = dt.columns
  time_transform = pd.DataFrame(dt.Id, index=dt.index)
  for sort in sorts:
    dt.sort_values(by=sort,inplace=True)
    name = ''.join(sort)
    for c in np.setdiff1d(cols,'Id'):
      time_transform[name+'_prevTimeDiff'] = dt[c].diff().fillna(9999999)
      time_transform[name+'_nextTimeDiff'] = dt[c].diff(-1).fillna(9999999)
      time_transform[name+'_PrevTime'] = dt[c].shift(1).fillna(9999999)
      time_transform[name+'_NexTime'] = dt[c].shift(-1).fillna(9999999)
  time_transform.set_index('Id', drop=True, inplace=True)
  del dt
  gc.collect()
  return time_transform

def time_shift(df,sorts):
  dt = df.copy()
  dt.reset_index(inplace=True)
  cols = dt.columns
  time_shift = pd.DataFrame(dt.Id, index=dt.index)
  for sort in sorts:
      dt.sort_values(by=sort,inplace=True)
      name = ''.join(sort)
      for c in [cols,'Id']:
        time_shift[name+'_PrevTime'] = dt[c].shift(1).fillna(9999999)
        time_shift[name+'_NexTime'] = dt[c].shift(-1).fillna(9999999) 
  time_shift.set_index('Id', drop=True, inplace=True)
  del dt
  gc.collect()
  return time_shift

def period_count_shift(df, suffix, period=16.8,shift=1):          
  dt = df.copy()
  cols = dt.columns.tolist()
  for f in cols:
      tmp_count = (dt[f] / period) // 1.0   
      tmp_group = tmp_count.groupby(tmp_count).count()
      shhiftdown = tmp_group.shift(shift).fillna(0)
      shhiftup = tmp_group.shift(-shift).fillna(0)
      dt[f+'_next'+suffix] = tmp_count.map(shhiftdown)
      dt[f+'_prev'+suffix] = tmp_count.map(shhiftup)
  dt.drop(cols, axis=1, inplace=True)
  return dt

def time_bins(df,bin_edges=None): 
  time_bins = {}
  time_binned = {}
  for f in tqdm(df.columns):
      if not bin_edges:
        # if bins are not provided, use quantile cut
        bins = int(max(10, df[f].notnull().sum()/20000))
        time_binned[f+'_bin'], time_bins[f] = pd.qcut(df[f], retbins=True,
            q=bins, labels=False, duplicates='drop')
      else:
        # if bin edges are provided, use cut
        time_binned[f+'_bin'], time_bins[f] = pd.cut(df[f], retbins=True,
            bins=bin_edges[f], labels=False, duplicates='drop')         
  time_binned = pd.DataFrame(time_binned) 
  return  time_binned, time_bins

def time_modulo(df, suffix='_mod_week', period=16.8):  
  dt = df.copy()
  cols = dt.columns.tolist()
  for c in cols:
      dt[c+suffix] = dt[c] % period
  dt.drop(cols, axis=1, inplace=True) 
  return dt

def time_all_feats(df):
  time_all = pd.DataFrame(index=df.index)
  time_all['all_start'] = df.min(axis=1) 
  time_all['all_end'] = df.max(axis=1)
  time_all['all_duration'] = time_all['all_end'] - time_all['all_start'] 
  print("time_all_feats starting......")
  # iddiff
  sorts = ['all_start','all_end']
  idDiff_feat = id_diff(time_all, sorts).astype(int)
  # time_diff
  sorts = ['Id',['all_start','Id'],['all_end','Id'],['all_duration','Id'],['all_start','all_end'],'idx']
  time_transform_feat = time_transform(time_all,sorts).astype(int)
  # time modulo
  feats = ['all_start','all_end']
  dt = time_all[feats]
  time_in_qd = time_modulo(dt, suffix='_mod_qd', period=0.25)
  time_in_day = time_modulo(dt, suffix='_mod_day', period=2.4)
  time_in_week = time_modulo(dt, suffix='_mod_week', period=16.8)
  # col value counts
  time_value_cnt = time_values_count(time_all, feats)
  # period count shift
  period_count_shift_week = period_count_shift(dt, '_p1w', 16.8)  
  period_count_shift_day = period_count_shift(dt, '_p1d', 2.4) 
  period_count_shift_qd = period_count_shift(dt, '_p1qd', 0.25) 

  time_all_feats = pd.concat([time_all, idDiff_feat, time_transform_feat, time_in_qd, time_in_day , time_in_week,
                     time_value_cnt, period_count_shift_week, period_count_shift_day , period_count_shift_qd],axis=1)
  save_pickle(time_all_feats,os.path.join(featuresdir,'time_all_feats.pickle'))
  del time_all, idDiff_feat, time_transform_feat, time_in_qd, time_in_day ,\
     time_in_week, time_value_cnt, period_count_shift_week, period_count_shift_day , period_count_shift_qd
  gc.collect()
  print("time_all_feats ending......")
  return time_all_feats

def line_time_feats(df): 
  import re
  dt = multiIndex_cols(df,'L')
  dt = time_extract(dt,'L')
  print("line_time_feats starting......")
  # iddiff
  sorts = dt.columns
  idDiff_feat = id_diff(dt, sorts).astype(int)
  # binned
  line_time_binned, _ = time_bins(dt)
  # modelo
  feats = re.findall(r'L[0-5]+_start|L[0-5]+_end',''.join(dt.columns))
  tem = dt[feats]
  time_in_qd = time_modulo(tem, suffix='_mod_qd', period=0.25)
  time_in_day = time_modulo(tem, suffix='_mod_day', period=2.4)
  time_in_week = time_modulo(tem, suffix='_mod_week', period=16.8)
  # col value counts
  cols_value_cnt = time_values_count(dt, feats)
  # period count shift
  period_count_shift_week = period_count_shift(tem, '_p1w', 16.8)  
  period_count_shift_day = period_count_shift(tem, '_p1d', 2.4) 
  period_count_shift_qd = period_count_shift(tem, '_p1qd', 0.25) 
  
  line_time_feats  = pd.concat([dt, idDiff_feat, line_time_binned, time_in_qd, time_in_day, time_in_week, cols_value_cnt,
                     period_count_shift_week,period_count_shift_day,period_count_shift_qd],axis=1)
  save_pickle(line_time_feats,os.path.join(featuresdir,'line_time_feats.pickle'))
  del dt, idDiff_feat, line_time_binned, time_in_qd, time_in_day, time_in_week, cols_value_cnt,\
                     period_count_shift_week,period_count_shift_day,period_count_shift_qd
  gc.collect()
  print("line_time_feats ending......") 
  return   line_time_feats

def station_time_feats(df):
  dt = multiIndex_cols(df,'S')
  dt = time_extract(dt,'S')
  dt = station_time_filter(dt)
  print("station_time_feats starting......")
  # diff
  sorts = dt.columns
  idDiff_feat = id_diff(dt, sorts).astype(int)
  # binned
  station_time_binned, _ = time_bins(dt)
  station_time_feats = pd.concat([dt,idDiff_feat,station_time_binned],axis=1)
  save_pickle(station_time_feats,os.path.join(featuresdir,'station_time_feats.pickle'))
  del dt, idDiff_feat, station_time_binned
  gc.collect()
  print("station_time_feats ending......")
  return station_time_feats

def z_score(df):
  dfz = {}
  cols = df.columns
  print('Calculating z-scores:')
  for c in tqdm(cols):
      dfz[c+'_z'] = (df[c] - df[c].mean()) / df[c].std()
  dfz = pd.DataFrame(dfz)
  return dfz

def sample_counts(df):
  dfc = {}
  cols = df.columns
  print('Calculating sample counts:')
  for c in tqdm(cols):
      dfc[c+'_c'] = df[c].map(df[c].groupby(df[c]).count())
  dfc = pd.DataFrame(dfc) 
  return dfc

def important_numeric_feats(X,y):
  from xgboost import XGBClassifier, plot_importance
  clf = XGBClassifier(max_depth=9, n_estimators=100, base_score=0.0058, n_jobs=4, colsample_bytree=0.6,
             min_child_weight=5, subsample=0.9,  reg_lambda=4, silent=False, learning_rate=0.03)
  clf.fit(X, y, eval_set=[(X, y)], eval_metric='auc', verbose=True)  
  feats_filer_index = np.where(clf.feature_importances_>0.002)[0] 
  important_feats = X.columns[feats_filer_index]  
  important_feats = pd.Series(important_feats)
  important_feats.to_csv(os.path.join(featuresdir,'important_numeric_feats.csv'))
  fig, ax = plt.subplots(figsize=(12,18))
  plot_importance(clf, max_num_features=75, height=0.8, ax=ax,importance_type='gain', show_values = False)
  # plt.plot(clf.feature_importances_[feats_filer_index].cumsum())
  return important_feats,clf

def get_numeric_feats(X,y,prefix):
  important_feats,_ = important_numeric_feats(X,y)
  numeric_feats = X[important_feats]
  numeric_feats_z = z_score(numeric_feats)
  numeric_feats_c = sample_counts(numeric_feats)
  numeric_feats_transformed = pd.concat([numeric_feats,numeric_feats_z, numeric_feats_c], axis=1)
  save_pickle(numeric_feats_transformed, os.path.join(featuresdir,prefix+'_feats.pickle'))  
  del numeric_feats, numeric_feats_z, numeric_feats_c
  gc.collect()  
  return numeric_feats_transformed


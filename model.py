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

X_train = read_pickle('train_all_feats.pickle')
X_test = read_pickle('test_all_feats.pickle')
param_grid = {'gamma': [0,0.1,0.4,0.8,1.6,3.2,6.4],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.4, 0.7],
        'max_depth': [5,8,10,15,20],
        'n_estimators': [50,80,100,130,150],
        'reg_alpha': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
        'reg_lambda': [0,0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2,102.4,200],
        "min_child_weight" : [ 1, 3, 5, 7],
        "colsample_bytree" : [ 0.3, 0.5 , 0.7]}
clf = XGBClassifier()
y = get_response()
best_params = random_search_report(clf,X_train,y,param_grid)
clf = XGBClassifier(best_params)
mccs, aucs, threshs = cross_val_predict_tresh(clf,X_train,y)
make_submission(clf,X_test,best_threshold=np.mean(threshs))


# functions
import pickle
def save_pickle(x, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
def read_pickle(filename):
    with open(filename, 'rb') as handle:
        x = pickle.load(handle)
    return x
# hyperparameter research
def random_search_report(clf, X, y, param_grid, n_iter=4, n_splits=3, n_repeats=2):
  from sklearn.model_selection import RandomizedSearchCV, RepeatedStratifiedKFold
  from sklearn.metrics import make_scorer, matthews_corrcoef, roc_auc_score
  scorer = make_scorer([matthews_corrcoef, roc_auc_score])
  rskf = RepeatedStratifiedKFold(n_repeats=n_repeats, n_splits=n_splits, random_state=0)
  random_search = RandomizedSearchCV(clf, param_grid, scoring=scorer, n_iter=n_iter, cv=rskf, random_state=999, verbose=3)
  random_search.fit(X,y,eval_metric = 'auc')
  print("Random search for estimator : ", clf)
  print("Best score: ", random_search.best_score_)
  print("Best params: ", random_search.best_params_) 
  return random_search.best_params_

# validation and threshold selection
def cross_val_predict_tresh(clf, X, y, n_splits=3, n_repeats=2):
  rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=444) 
  n_folds = n_splits * n_repeats
  preds = np.ones(y.shape[0])
  aucs = np.ones(n_folds)
  mccs = np.ones(n_folds)
  threshs = np.ones(n_folds)
  fig, ax = plt.subplots(2, 2, figsize=(8,6))
  for i, (train, test) in enumerate(rskf.split(X, y)):
    print("Fold", i)
    clf.fit(X[train], y[train], 
        eval_set=[(X[test], y[test])], 
        eval_metric='auc', 
        random_state=333,
        early_stopping_rounds=30,
        verbose=3)
#   print("fold {}, best round: {}".format(i, clf_test.best_round))
    preds[test] = clf.predict_proba(X[test])[:,1]
    print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(X[test], preds[test])))
    thresholds = np.linspace(0.01, 0.5, 10)
    mcc = np.array([matthews_corrcoef(y[test], preds[test]>thr) for thr in thresholds])  
    print("fold {}, TH: {:.3f}, MCC: MAX {:.3f}".format(i,thresholds[mcc.argmax()],mcc.max()))
    threshs[i] = thresholds[mcc.argmax()]
    aucs[i] = roc_auc_score(y[test], preds[test])
    mccs[i] = mcc.max()
  print("mean out of fold AUC "+str(np.mean(aucs)))
  print("mean out of fold MCC "+str(np.mean(mccs)))
  print("mean out of fold ths "+str(np.mean(threshs)))
  return mccs, aucs, threshs

def plot_roc(y_test, pred):
  fpr, tpr, threshold = roc_curve(y_test, pred)
  roc_auc = roc_auc_score(fpr, tpr)
  plt.figure(figsize=(6,6))
  plt.title('Validation ROC')
  plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
  plt.legend(loc = 'lower right')
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()

def make_submission(clf,X_test,name='my_submission.csv.zip',best_threshold=0.29):
  preds = (clf.predict_proba(X_test)[:,1] > best_threshold).astype(np.int8)
  sub = pd.read_csv(os.path.join(datadir,'sample_submission.csv'), index_col=0)
  sub["Response"] = preds
  sub.to_csv(name, compression="zip")
  print(name+" saved!")
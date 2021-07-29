# -*- coding: utf-8 -*-
import sys
old_stdout = sys.stdout
# log_file = open("message.log","w")
# sys.stdout = log_file
print("Hello")

from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns
from numpy.random import seed
import numpy as np
np.random.seed(0)

# https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas
def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n=10):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

def z_score(df):
    # copy the dataframe
    df_std = df.copy()
    # apply the z-score method
    for column in df_std.columns:
        df_std[column] = (df_std[column] - df_std[column].mean()) / df_std[column].std()
        
    return df_std

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def calc_f1_score(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return (2*((precision*recall)/(precision+recall+K.epsilon()))).numpy(), precision.numpy(), recall.numpy()

imp = pd.read_csv("x.csv")

# imp = pd.read_csv("x.csv")
imp.entrance_unz_date = pd.to_datetime(imp.entrance_unz_date)
imp.ort_vor_aufnahme_spital = pd.factorize(imp.ort_vor_aufnahme_spital)[0]
imp.referral_unz = pd.factorize(imp.referral_unz)[0]
imp.admission_choice = pd.factorize(imp.admission_choice)[0]
imp_x = imp.sort_values(by='entrance_unz_date')
imp_x.entrance_unz_date = pd.to_datetime(imp_x.entrance_unz_date).dt.dayofyear

to_predict = "blood_culture_positive" # "mort_28_days" # 

print(len(imp_x))
imp_x.dropna(axis=0, subset=[to_predict], inplace=True)
print(len(imp_x))
proc_y = imp_x[[to_predict]]

features = [to_predict, "pseudoid_patient", "age_admission_unz", "sex", "triage", "referral_unz", "ort_vor_aufnahme_spital", "time_unz", "admission_choice",
                 "pulse_first", "frequency_first", "systolic_bp_first",
                 "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk", "entrance_unz_date"]

to_check = imp_x[features]

for i in to_check.columns[1:]:
  for j in range(1,10):
    to_check[i+"inf"+str(j)] = (to_check[i] >= to_check[i].quantile(0.1*j)).astype(int)
    # print(to_check[i+"inf"+str(j)])
imp_x = to_check
to_check = to_check[[to_predict, "pseudoid_patient", "LACT =lactat", "LACT =lactatinf7", "CRP", "GGT", "GGTinf5", "INRiHinf5", "INRiH", "THZ (Thrombozyten)", "THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoff", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "BIC_VB",  "BIC_st",  "BIC_stinf1", "lvl_consc_alert",
                   "pHinf1", "CRinf8", "EOS =eosinophileinf6", "age_admission_unz", "diastolic_bp_firstinf4", "respiratory_rate_firstinf3", "systolic_bp_firstinf4", "Hb", "PCThs"]]

corrMatrix = to_check.corr()
print(corrMatrix)
corrMatrix.to_csv('corrMatrix.csv')
get_top_abs_correlations(corrMatrix, n=30)

features1 = [to_predict, "pseudoid_patient", "age_admission_unz", "sex", "triage", "referral_unz", "ort_vor_aufnahme_spital", "time_unz", "admission_choice",
                 "pulse_first", "frequency_first", "systolic_bp_first", "systolic_bp_first_inf_100",
                 "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk", "leuk_sup_10", "leuk_inf_4", "leuk_inf_0_5", "entrance_unz_date"]

features2= [to_predict, "pseudoid_patient", "age_admission_unz", "sex", "triage", "referral_unz", "ort_vor_aufnahme_spital", "admission_choice",
                 "pulse_first", "frequency_first", "systolic_bp_first",
                 "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first",
                 "lvl_consc_alert", "spo2_first", "temperature_highest",
                 "o2_gabe", "THZ (Thrombozyten)", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "CRP", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk",]

features3 = [to_predict, "pseudoid_patient", "age_admission_unz", "sex", "respiratory_rate_first", "THZ (Thrombozyten)", "lvl_consc_alert", "o2_gabe", "INRiH", "Hb", "UREA = Harnstoff", "CRP", "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB"]

features4 = [to_predict, "pseudoid_patient", "age_admission_unz", "THZ (Thrombozyten)" , "CRP", "LACT =lactat", "BIC_st", "ASAT"]

features5 = [to_predict, "pseudoid_patient", "age_admission_unz", "CRP", "PCThs"]

features_better = [to_predict, "pseudoid_patient","LACT =lactat", "LACT =lactatinf7", "CRP", "GGT", "GGTinf5",  "CRPinf8",  "INRiHinf5", "INRiH", "THZ (Thrombozyten)", "THZ (Thrombozyten)inf3", "ASAT", "ASATinf3", "UREA = Harnstoff", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "BIC_VB",  "BIC_st",  "BIC_stinf1", "lvl_consc_alert",
                   "pH", "pHinf1", "CR", "CRinf8", "EOS =eosinophile", "EOS =eosinophileinf6", "age_admission_unz", "diastolic_bp_first", "diastolic_bp_firstinf4", "respiratory_rate_first", "respiratory_rate_firstinf3", "systolic_bp_first", "systolic_bp_firstinf4", "Hb", "PCThs", "KA = Kalium", "KA = Kaliuminf2", "pulse_first", "pulse_firstinf5", "NA", "NAinf2",
                   "gcs_inf_15_first"]

features_better2 = [to_predict, "LACT =lactat", "LACT =lactatinf7", "CRP", "GGT", "GGTinf5", "INRiHinf5", "INRiH", "THZ (Thrombozyten)", "THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoff", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "BIC_VB",  "BIC_st",  "BIC_stinf1", "lvl_consc_alert",
                   "pHinf1", "CRinf8", "EOS =eosinophileinf6", "age_admission_unz", "diastolic_bp_firstinf4", "respiratory_rate_firstinf3", "systolic_bp_firstinf4", "Hb", "PCThs"]

features_better3 = [to_predict, "pseudoid_patient", "LACT =lactat", "LACT =lactatinf7", "CRP","GGT", "GGTinf5","INRiHinf5","UREA = Harnstoff","BIC_st", "pH", "lvl_consc_alert", "PCThs"]
from collections import Counter


d =  Counter(features_better)  # -> Counter({4: 3, 6: 2, 3: 1, 2: 1, 5: 1, 7: 1, 8: 1})
res = [k for k, v in d.items() if v > 1]
print(res)


# feature_list = [features1, features2, features3, features4, features5]
# feature_list = [features5]
feature_list = [features_better2]
feature_name = ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"]

def print_results(y_pred, y_test, move=0.0):
  y_pred = pd.DataFrame(y_pred)
  y_pred = round(y_pred+move)
  y_pred = tf.cast(y_pred, tf.float32)
  y_test = tf.cast(y_test, tf.float32)
  f1=calc_f1_score(y_test, y_pred)


  y_pred = tf.cast(y_pred, tf.int32)
  y_test = tf.cast(y_test, tf.int32)
  accuracy = accuracy_score(y_test, y_pred)
  roc_auc = roc_auc_score(y_test, y_pred, sample_weight=None)
  print("F1 Score:", f1[0], "Precision:", f1[1], "Recall:", f1[2], "ROC_AUC", roc_auc, "Accuracy:", accuracy)

def get_f1(y_test, y_pred, move=0.0):
  y_pred = pd.DataFrame(y_pred)
  y_pred = round(y_pred+move)
  y_pred = tf.cast(y_pred, tf.float32)
  y_test = tf.cast(y_test, tf.float32)
  return calc_f1_score(y_test, y_pred)

from keras import models, optimizers
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def dense_nn(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Dense NN")
  model = models.Sequential()
  model.add(Dense(32, activation='relu', input_shape=(len(X_train.columns),)))
  model.add(Dropout(0.25))
  model.add(Dense(16, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(8, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer= optimizers.Adam(lr=0.01), metrics=['accuracy', f1])

  earlyStopping = EarlyStopping(monitor='val_f1', patience=10, verbose=0, mode='max')
  mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_f1', mode='max')
  reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')
  return model
  # Train model
  model.fit(X_train, y_train,
          batch_size=64,
          epochs=1024,
          verbose=1,
          validation_data=(X_val, y_val),
          callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
          class_weight = {0: 1., 1: class_weight})

  score = model.evaluate(X_test, y_test, verbose=0)
  y_pred = model.predict(X_test)
  print_results(y_pred, y_test, 0)

import xgboost as xgb

def xg_boost_classifier(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("XG Boost Classifier")
  xgb_model = xgb.XGBClassifier(objective="reg:squarederror", random_state=0, scale_pos_weight=class_weight, max_depth=16, learning_rate=0.1, n_estimators=200)
  xgb_model.fit(X_train, y_train.values.flatten())
  y_pred = xgb_model.predict(X_test)
  print_results(y_pred, y_test, 0)

import xgboost as xgb

def xg_boost_regressor(X_train, y_train, X_val, y_val, X_test, y_test):
  print("XG Boost Regressor")
  xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=0)
  xgb_model.fit(X_train, y_train.values.flatten())
  y_pred = xgb_model.predict(X_test)
  print_results(y_pred, y_test, 0)

from sklearn.linear_model import LinearRegression


def linear_regression(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Linear Regression")
  regressor = LinearRegression()
  regressor.fit(X_train, y_train, sample_weight = (class_weight-1)*y_train.values.flatten()+1) 
  y_pred = regressor.predict(X_test) 
  print_results(y_pred, y_test, 0)

from sklearn.tree import DecisionTreeClassifier

def decision_tree_class(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Decision Tree Classifier")
  regressor = DecisionTreeClassifier(criterion="gini", splitter="best", class_weight={0: 1, 1: class_weight}, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state = 0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort="deprecated", ccp_alpha=0.0) 
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test) 
  print_results(y_pred, y_test, 0)

from sklearn.tree import DecisionTreeRegressor 


def decision_tree_reg(X_train, y_train, X_val, y_val, X_test, y_test):
  print("Decision Tree Regressor")
  regressor = DecisionTreeRegressor(criterion="mse", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state = 0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort="deprecated", ccp_alpha=0.0) 
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test) 
  print_results(y_pred, y_test, 0)

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def svc_rbf(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("SVC RBF")
  clf = SVC(gamma='auto', random_state=0, kernel='rbf', class_weight={0: 1, 1: class_weight})
  clf.fit(X_train, y_train.values.flatten())
  y_pred = clf.predict(X_test)
  print_results(y_pred, y_test, 0)

from sklearn.ensemble import RandomForestClassifier

def random_forest_classifier(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Random Forest Classifier")
  rf = RandomForestClassifier(random_state=0, class_weight={0: 1, 1: class_weight})
  rf.fit(X_train, y_train.values.flatten())
  y_pred = rf.predict_proba(X_test)
  print_results(y_pred[:,1], y_test, 0)

from sklearn.ensemble import RandomForestRegressor

def random_forest_regressor(X_train, y_train, X_val, y_val, X_test, y_test):
  print("Random Forest Regression")
  rf = RandomForestRegressor(random_state=0)
  rf.fit(X_train, y_train.values.flatten())
  y_pred = rf.predict(X_test)
  print_results(y_pred, y_test, 0)

from sklearn.neural_network import MLPClassifier

def mlp_classifier(X_train, y_train, X_val, y_val, X_test, y_test):
  print("MLP Classifier")
  clf = MLPClassifier(random_state=0, max_iter=1000).fit(X_train, y_train.values.flatten())
  y_pred = clf.predict_proba(X_test)
  print_results(y_pred[:,1], y_test, 0)

from sklearn.linear_model import LassoCV
from sklearn.model_selection import StratifiedKFold

def lasso_cv(X_train, y_train, X_val, y_val, X_test, y_test):
  print("Lasso CV")
  pipe = LassoCV(cv=StratifiedKFold(n_splits=5))
  pipe.fit(X_train, y_train.values.flatten())
  y_pred = pipe.predict(X_test)
  print_results(y_pred, y_test, 0)

from sklearn.linear_model import RidgeClassifier

def ridge_classifier(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Ridge Classifier")
  clf = RidgeClassifier(alpha=1, class_weight={0: 1, 1: class_weight})
  clf.fit(X_train, y_train.values.flatten())
  y_pred = clf.predict(X_test)
  print_results(y_pred, y_test, 0)

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

def gaussian_process(X_train, y_train, X_val, y_val, X_test, y_test):
  print("Gaussian Process")
  kernel = 1.0 * RBF(1.0)
  gp = GaussianProcessClassifier(kernel=kernel, random_state=0)
  gp.fit(X_train, y_train)
  y_pred = gp.predict_proba(X_test)
  print_results(y_pred[:,1], y_test, 0)

from sklearn import linear_model
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import Pipeline
from sklearn.gaussian_process.kernels import Matern
from matplotlib import pyplot as plt

def gaussian_process_approx(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Gaussian Process Approximation")
  kernel_approximation = Nystroem(kernel=Matern(length_scale=1.0, nu=1.5), n_components=1000, random_state=0)
  model = linear_model.BayesianRidge()

  pipeline_steps = [
              ('kernelApproximation', kernel_approximation),
              ('model', model)
          ]
  pipeline = Pipeline(pipeline_steps)
  pipeline.fit(X_train, y_train.values.flatten(), model__sample_weight = (class_weight-1)*y_train.values.flatten()+1)
  y_pred, std = pipeline.predict(X_test, return_std=True)
  print(np.mean(std))
  plt.hist(std)
  print_results(y_pred, y_test, 0) 

# Baysian Regression Second arguement get standart diviation uncertainity

from sklearn.naive_bayes import GaussianNB

def naive_bayes(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Naive Bayes")
  gnb = GaussianNB()
  gnb.fit(X_train, y_train.values.flatten(), sample_weight = (class_weight-1)*y_train.values.flatten()+1)
  y_pred = gnb.predict(X_test)
  print_results(y_pred, y_test, 0)

from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier

def sgd_classifier(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("SGD Classifier")
  rbf_feature = RBFSampler(gamma=1, random_state=0)
  X_features = rbf_feature.fit_transform(X_train)
  clf = SGDClassifier(max_iter=100, tol=1e-3, class_weight={0: 1, 1: class_weight})
  clf.fit(X_features, y_train.values.flatten())
  X_test_trans = rbf_feature.fit_transform(X_test)
  y_pred = clf.predict(X_test_trans)
  print_results(y_pred, y_test, 0)

from sklearn import linear_model

def bayesian_ridge(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Bayesian Ridge")
  model = linear_model.BayesianRidge()
  model.fit(X_train, y_train.values.flatten(), sample_weight = (class_weight-1)*y_train.values.flatten()+1)
  y_pred = model.predict(X_test)
  print_results(y_pred, y_test, 0)

from keras import models, optimizers
from keras.layers import Dense, Dropout, RNN, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def bi_rnn(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Bidirectinal RNN")
  model = models.Sequential()
  model.add(Bidirectional(RNN(32, return_sequences=True), input_shape=(len(X_train.columns),)))
  model.add(Bidirectional(RNN(16)))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer= optimizers.Adam(lr=0.01), metrics=['accuracy', f1, 'auc_roc'])

k=0
for i in feature_list:
  proc_x = imp_x[i]
  proc_x = proc_x.fillna(-1)
  proc_x = z_score(proc_x)

  corrMatrix = proc_x.corr()
  print(corrMatrix)
  print(get_top_abs_correlations(corrMatrix))

  pca = PCA(n_components=2)
  pca.fit(proc_x[proc_x.columns[1:]])
  print(pca.explained_variance_ratio_)
  print(pca.singular_values_)
  proc_x = proc_x[proc_x.columns[1:]]
  """
  X_embedded = TSNE(n_components=2, random_state=0).fit_transform(proc_x)
  df = pd.DataFrame()
  df["y"] = proc_y.values.flatten()
  df["comp1"] = X_embedded[:,0]
  df["comp2"] = X_embedded[:,1]
  sns.scatterplot(x="comp1", y="comp2", hue=df.y.tolist(),
        palette=sns.color_palette("hls", 2),
        data=df).set(title=feature_name[j]+" data T-SNE projection")

  plt.savefig(feature_name[j]+".png")
  """
  k+=1

  X_train, X_test, y_train, y_test = train_test_split(proc_x, proc_y, test_size=0.2, shuffle=False)
  # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)
  X_train, _, y_train,_ = train_test_split(X_train, y_train, test_size=0.0001, random_state=0, shuffle=True)
  # X_val, _, y_val,_ = train_test_split(X_val, y_val, test_size=0.0001, random_state=0, shuffle=True)
  X_val = ""
  y_val = ""
  
  # display(X_train)

  # for j in i[1:]:
  #  X_train_old = X_train.copy()
  #  X_train = X_train_old[np.abs(X_train_old[j]-X_train_old[j].mean())<=(3*X_train_old[j].std())]
  #  y_train = y_train[np.abs(X_train_old[j]-X_train_old[j].mean())<=(3*X_train_old[j].std())]
  bin_count = np.bincount(y_train.values.flatten())
  class_weight = bin_count[0]/bin_count[1]
 
  # dense_nn(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  
  # xg_boost_classifier(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  # xg_boost_regressor(X_train, y_train, X_val, y_val, X_test, y_test)
  linear_regression(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  """
  decision_tree_class(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  decision_tree_reg(X_train, y_train, X_val, y_val, X_test, y_test)
  svc_rbf(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  random_forest_classifier(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  random_forest_regressor(X_train, y_train, X_val, y_val, X_test, y_test)
  mlp_classifier(X_train, y_train, X_val, y_val, X_test, y_test)
  lasso_cv(X_train, y_train, X_val, y_val, X_test, y_test) 
  """
  ridge_classifier(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  gaussian_process_approx(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  #naive_bayes(X_train, y_train, X_val, y_val, X_test, y_test, class_weight) 
  #sgd_classifier(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  #bayesian_ridge(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)
  
  # bi_rnn(X_train, y_train, X_val, y_val, X_test, y_test, class_weight)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


print(feature_list[0])
print(len(feature_list[0]))
proc_x = imp_x[feature_list[0]]
proc_x = proc_x.fillna(-1)
proc_x = z_score(proc_x)
proc_x = proc_x[proc_x.columns[2:]]
proc_x = proc_x + 1
proc_x[proc_x < 0] = 0
print(proc_x)
selector = SelectKBest(chi2, k=10)
X_new = selector.fit_transform(proc_x, proc_y)
print(X_new.shape)
print(X_new)
index_best = selector.get_support(indices=True)
print(index_best)
for i in index_best:
  print(feature_list[0][i+1])

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingClassifier

proc_y = imp_x[[to_predict]]

# proc_y = df["blood_culture_positive"]

for q in feature_list:
  print(q)
  proc_x = imp_x[q]
  proc_x = proc_x.fillna(-1)
  proc_x = z_score(proc_x)
  proc_x = proc_x[proc_x.columns[2:]]
  X_train, X_test, y_train, y_test = train_test_split(proc_x, proc_y, test_size=0.2, shuffle=False)
  # X_train, _, y_train,_ = train_test_split(X_train, y_train, test_size=0.0001, random_state=0, shuffle=True)
  bin_count = np.bincount(y_train.values.flatten())
  class_weight = bin_count[0]/bin_count[1]
  """
  clf = RidgeClassifier(alpha=1, class_weight={0: 1, 1: class_weight})
  cv_results = cross_validate(clf, X_train, y_train, cv=5, scoring=('roc_auc'))

  kf = KFold(n_splits=10, random_state=None, shuffle=False)

  for i in range(0):
    for train_index, test_index in kf.split(X_train):
      print("TRAIN:", train_index, "TEST:", test_index)
      X_trains, X_tests = X_train.iloc[train_index], X_train.iloc[test_index]
      y_trains, y_tests = y_train.iloc[train_index], y_train.iloc[test_index]
      clf = RidgeClassifier(alpha=1, class_weight={0: 1, 1: class_weight}, random_state=0)
      clf.fit(X_trains, y_trains)
      y_preds = clf.predict(X_tests)
      print_results(y_preds, y_tests, 0)
  """
  def get_gauss_approx():
    kernel_approximation = Nystroem(kernel=Matern(length_scale=1.0, nu=1.5), n_components=1000, random_state=0)
    model = linear_model.BayesianRidge(n_iter=300)

    pipeline_steps = [
                  ('kernelApproximation', kernel_approximation),
                  ('model', model)
              ]
    return Pipeline(pipeline_steps)
  
  kf = GroupKFold(n_splits=8)
  print(imp_x.columns)
  print(proc_x)
  # models =[LinearRegression , Ridge, Pipeline, RidgeClassifier, linear_model.BayesianRidge, DecisionTreeRegressor, xgb.XGBRegressor, GaussianNB, SGDClassifier, RandomForestClassifier, RandomForestRegressor]
  models =[LinearRegression , Ridge, RidgeClassifier, linear_model.BayesianRidge, xgb.XGBClassifier, GaussianNB]
  
  args = [{}, {}, {'class_weight': 'balanced'}, {}, {}, {}, {'class_weight': 'balanced'}, {}, {}, {}, {}, {}, {}]
  # models =[LinearRegression , Ridge, Pipeline]
  for k in range(len(models)):
    roc_auc_list = []
    f1_list = []
    for train_index, test_index in kf.split(proc_x, proc_y, groups=imp_x["pseudoid_patient"]):
      if False: # k==2:
        clf = get_gauss_approx()
      else:
        clf = models[k](**args[k])
      
      X_trains, X_tests = proc_x.iloc[train_index], proc_x.iloc[test_index]
      y_trains, y_tests = proc_y.iloc[train_index], proc_y.iloc[test_index]
      if k==2:
        clf.fit(X_trains, y_trains.values.flatten())
      else:
        clf.fit(X_trains, y_trains.values.flatten(), sample_weight = (class_weight-1)*y_trains.values.flatten()+1)
      y_preds = list(map(round, clf.predict(X_tests)))
      
      roc_auc_list.append(roc_auc_score(y_tests, y_preds, sample_weight=None))
      f1_list.append(get_f1(y_tests, y_preds))
    print(clf, "GrAnd Total:", np.mean(roc_auc_list), np.mean(f1_list))
"""
print("For Test")
clf = LinearRegression()
dfasd = cross_val_score(clf, X_train, y_train, scoring="roc_auc", cv=GroupKFold)
print(dfasd, np.mean(dfasd))

# clf.fit(X_train, y_train)
print("Whole Dataset")
y_pred = clf.predict(X_test)
print_results(y_pred, y_test, 0)
"""

from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import VotingClassifier

for q in feature_list:
  print(q)
  proc_x = imp_x[q]
  proc_x = proc_x.fillna(-1)
  proc_x = z_score(proc_x)
  proc_x = proc_x[proc_x.columns[2:]]
  bin_count = np.bincount(y_train.values.flatten())
  class_weight = bin_count[0]/bin_count[1]

  
  kf = GroupKFold(n_splits=10)
  models =[LinearRegression , Ridge, RidgeClassifier, linear_model.BayesianRidge, xgb.XGBClassifier, GaussianNB]
  
  args = [{}, {}, {'class_weight': 'balanced'}, {}, {}, {}, {'class_weight': 'balanced'}, {}, {}, {}, {}, {}, {}]
  roc_auc_list = []
  f1_list = []
  for train_index, test_index in kf.split(proc_x, proc_y, groups=imp_x["pseudoid_patient"]):
    clf1 = LinearRegression()
    clf2 = Ridge()
    clf3 = RidgeClassifier(class_weight= 'balanced')
    clf4 = linear_model.BayesianRidge()
    clf5 = xgb.XGBClassifier()
    clf6 = GaussianNB()
    
    eclf = VotingClassifier(estimators=[('xgb', clf5), ('gnb', clf6)], voting='soft')

    X_trains, X_tests = proc_x.iloc[train_index], proc_x.iloc[test_index]
    y_trains, y_tests = proc_y.iloc[train_index], proc_y.iloc[test_index]
    
    clf3.fit(X_trains, y_trains.values.flatten())

    clf1.fit(X_trains, y_trains.values.flatten(), sample_weight = (class_weight-1)*y_trains.values.flatten()+1)
    clf2.fit(X_trains, y_trains.values.flatten(), sample_weight = (class_weight-1)*y_trains.values.flatten()+1)
    clf3.fit(X_trains, y_trains.values.flatten(), sample_weight = (class_weight-1)*y_trains.values.flatten()+1)
    clf4.fit(X_trains, y_trains.values.flatten(), sample_weight = (class_weight-1)*y_trains.values.flatten()+1)
    clf5.fit(X_trains, y_trains.values.flatten(), sample_weight = (class_weight-1)*y_trains.values.flatten()+1)

    eclf = eclf.fit(X_trains, y_trains.values.flatten())

    y_preds = list(map(round, eclf.predict(X_tests)))
    
    roc_auc_list.append(roc_auc_score(y_tests, y_preds, sample_weight=None))
    f1_list.append(get_f1(y_tests, y_preds))
    print(clf, "GrAnd Total:", np.mean(roc_auc_list), np.mean(f1_list))

from sklearn.tree import DecisionTreeClassifier

def decision_tree_class(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Decision Tree Classifier")
  regressor = DecisionTreeClassifier(criterion="gini", splitter="best", class_weight={0: 1, 1: class_weight}, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state = 0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort="deprecated", ccp_alpha=0.0) 
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test) 
  print_results(y_pred, y_test, 0)

from sklearn.tree import DecisionTreeClassifier

def decision_tree_class(X_train, y_train, X_val, y_val, X_test, y_test, class_weight):
  print("Decision Tree Classifier")
  regressor = DecisionTreeClassifier(criterion="gini", splitter="best", class_weight={0: 1, 1: class_weight}, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state = 0, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort="deprecated", ccp_alpha=0.0) 
  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test) 
  print_results(y_pred, y_test, 0)


from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score, f1_score
import xgboost as xgb
import numpy as np
import pandas as pd
np.random.seed(0)


cols = ["age_admission_unz", "triage", "referral_unz", "admission_choice",
"systolic_bp_first", "systolic_bp_first_inf_100", "diastolic_bp_first", 
"respiratory_rate_first", "gcs_inf_15_first", "lvl_consc_alert", 
"temperature_lowest", "spo2_first", "THZ (Thrombozyten)", "NA", "ASAT", 
"UREA = Harnstoff", "GFR", "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", 
"Hb", "INRiH", "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "leuk_sup_10", 
"leuk_inf_4", "leuk_inf_0_5", "LACT =lactatinf7", "GGTinf5", "INRiHinf5", 
"THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoffinf3", 
"ort_vor_aufnahme_spitalinf2", "BIC_stinf1",  "pHinf1", "CRinf8", 
"EOS =eosinophileinf6","diastolic_bp_firstinf4", "respiratory_rate_firstinf3", 
"systolic_bp_firstinf4", "NCH", "PLWS", "PSYCH", 'Hausarzt (privat)',
'Inselklinik', 'Polizei (In Begleitung)']

cols2 = ["age_admission_unz", "sex", "triage", "referral_unz", "ort_vor_aufnahme_spital", "time_unz", "admission_choice",
                 "pulse_first", "frequency_first", "systolic_bp_first", "systolic_bp_first_inf_100",
                 "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk", "leuk_sup_10", "leuk_inf_4", "leuk_inf_0_5", "entrance_unz_date"]

cols3 = ["age_admission_unz", "sex", "triage",
                 "pulse_first", "frequency_first", "systolic_bp_first",
                 "diastolic_bp_first", "respiratory_rate_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk"]

df = pd.read_csv("x.csv")

proc_y = df["blood_culture_positive"]

group_by = df["pseudoid_patient"]

df = df[cols2]

for i in cols3:
  for j in cols3:
    df[i+"div"+j] = df[i]/df[j]
    df[i+"mul"+j] = df[i]*df[j]

df = df.join(pd.get_dummies(df["admission_choice"]))
df = df.join(pd.get_dummies(df["referral_unz"]))
df = df.join(pd.get_dummies(df["ort_vor_aufnahme_spital"]))
df.referral_unz = pd.factorize(df.referral_unz)[0]
df.admission_choice = pd.factorize(df.admission_choice)[0]
df.ort_vor_aufnahme_spital = pd.factorize(df.ort_vor_aufnahme_spital)[0]
df = df.select_dtypes(['number'])

for i in cols3+["ort_vor_aufnahme_spital"]:
  for j in range(1,10):
    df[i+"inf"+str(j)] = (df[i] >= df[i].quantile(0.1*j)).astype(int)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


proc_x = df.copy(deep=True)
proc_x = proc_x.fillna(0)
proc_x[proc_x < 0.09] = 0
proc_x[proc_x > 100000] = 100000

for i in proc_x.columns:
  cor = proc_x[i].corr(proc_y)
  if abs(cor) > 0.1:
    print(i, cor)
selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(proc_x, proc_y)
index_best = selector.get_support(indices=True)

new_columns = map(lambda x: proc_x.columns[x], index_best)
proc_x = proc_x[new_columns]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

proc_x = pd.DataFrame(scaler.fit_transform(proc_x))


proc_x = df[cols] # .join(proc_x)
proc_x = proc_x.fillna(0)
proc_x[proc_x < 0.09] = 0

hp = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }


kf = GroupKFold(n_splits=8)
roc_auc_list = []
round_roc_auc = []
f1_list = []

for train_index, test_index in kf.split(proc_x, proc_y, groups=group_by):
    X_train, X_test = proc_x.iloc[train_index], proc_x.iloc[test_index]
    y_train, y_test = proc_y.iloc[train_index], proc_y.iloc[test_index] 
    # clf = xgb.XGBClassifier(objective="reg:squarederror", random_state=0, scale_pos_weight=5)
    # clf = RidgeClassifier(class_weight={0: 1, 1: 5})
    clf = LinearRegression()
    clf.fit(X_train, y_train, sample_weight = 2*y_train+1)
    y_pred = clf.predict(X_test)
    y_pred[y_pred < 0] = 0 
    y_pred[y_pred > 1] = 1
    count1=0
    count0=0 
    for i in y_pred[:30]:
      print(round(i,2), end=", ")
      
    print()
    print("Pred Quote", round(sum(np.round(y_pred, 0))/len(y_pred),2), "Train Quote", round(sum(np.round(y_train,0))/len(y_train),2))
    
    roc_auc_list.append(roc_auc_score(y_test, y_pred))
    round_roc_auc.append(roc_auc_score(y_test, np.round(y_pred+0.1,0)))
    f1_list.append(f1_score(y_test, np.round(y_pred+0.1,0)))

print(np.mean(roc_auc_list), np.mean(round_roc_auc), np.mean(f1_list)) # 0.6886 # 0.73

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge, RidgeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import xgboost as xgb
import numpy as np
import pandas as pd
np.random.seed(0)


cols = ["age_admission_unz", "triage", "referral_unz", "admission_choice",
"systolic_bp_first", "systolic_bp_first_inf_100", "diastolic_bp_first", 
"respiratory_rate_first", "gcs_inf_15_first", "lvl_consc_alert", 
"temperature_lowest", "spo2_first", "THZ (Thrombozyten)", "NA", "ASAT", 
"UREA = Harnstoff", "GFR", "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", 
"Hb", "INRiH", "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "leuk_sup_10", 
"leuk_inf_4", "leuk_inf_0_5", "LACT =lactatinf7", "GGTinf5", "INRiHinf5", 
"THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoffinf3", 
"ort_vor_aufnahme_spitalinf2", "BIC_stinf1",  "pHinf1", "CRinf8", 
"EOS =eosinophileinf6","diastolic_bp_firstinf4", "respiratory_rate_firstinf3", 
"systolic_bp_firstinf4", "NCH", "PLWS", "PSYCH", 'Hausarzt (privat)',
'Inselklinik', 'Polizei (In Begleitung)']

cols2 = ["age_admission_unz", "sex", "triage", "referral_unz", "ort_vor_aufnahme_spital", "time_unz", "admission_choice",
                 "pulse_first", "frequency_first", "systolic_bp_first", "systolic_bp_first_inf_100",
                 "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk", "leuk_sup_10", "leuk_inf_4", "leuk_inf_0_5", "entrance_unz_date"]

cols3 = ["age_admission_unz", "sex", "triage",
                 "pulse_first", "frequency_first", "systolic_bp_first",
                 "diastolic_bp_first", "respiratory_rate_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk"]


new_features = ["age_admission_unzmulLACT =lactat", "CRPdivBIC_VB", "LACT =lactatdivKA = Kalium", "BIC_stdivBIC_VB", "LACT =lactatinf7", "GGTmulCR", "frequency_firstdivBIC_st", "NAmulGGT", "GGTinf5", "UREA = HarnstoffdivKA = Kalium", "CRPinf8", "INRiHinf5", "NAdivBIC_VB", "ASATinf5", "UREA = Harnstoff", "frequency_firstdivpH", "GGT", "LACT =lactatinf3", "BIC_st", "pH", "ort_vor_aufnahme_spitalinf2", "temperature_highestinf7", "CR"]


cols77 = ['CRPdivBIC_VB', 'LACT =lactatdivKA = Kalium', 'CRmulGGT', 'INRiHinf5', 'THZ (Thrombozyten)inf3', 'lvl_consc_alertmulCRP', 'ort_vor_aufnahme_spitalinf2', 'temperature_highestinf7', 'CRPdivLeuk', 'GGTdivEOS =eosinophile', 'EOS =eosinophileinf6', 'Hbdivage_admission_unz', 'diastolic_bp_firstinf4', 'BIC_stdivTHZ (Thrombozyten)', 'systolic_bp_first_inf_100', 'KA = Kaliuminf2', 'triagemultriage', 'NEURO', 'frequency_firstdivPCThs', 'INRiHdivage_admission_unz', 'sexdivlvl_consc_alert', 'Hausarzt (privat)', 'sexdivdiastolic_bp_first', 'Krankenheim', 'temperature_lowestdivtemperature_highest', 'o2_gabedivASAT', 'Psych. Klinik', 'o2_gabedivtemperature_lowest', 'TCH'] #, 'VCH', 'SRMED']

df = pd.read_csv("x.csv")

proc_y = df["blood_culture_positive"]
group_by = df["pseudoid_patient"]

df = df[cols2]

for i in cols3:
  for j in cols3:
    df[i+"div"+j] = df[i]/df[j]
    df[i+"mul"+j] = df[i]*df[j]

df = df.join(pd.get_dummies(df["admission_choice"]))
df = df.join(pd.get_dummies(df["referral_unz"]))
df = df.join(pd.get_dummies(df["ort_vor_aufnahme_spital"]))
df = df.join(pd.get_dummies(df["triage"]))
df.referral_unz = pd.factorize(df.referral_unz)[0]
df.admission_choice = pd.factorize(df.admission_choice)[0]
df.ort_vor_aufnahme_spital = pd.factorize(df.ort_vor_aufnahme_spital)[0]

df = df.select_dtypes(['number'])

for i in cols3+["ort_vor_aufnahme_spital"]:
  for j in range(1,10):
    df[i+"inf"+str(j)] = (df[i] >= df[i].quantile(0.1*j)).astype(int)
"""
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


proc_x = df
proc_x = proc_x.fillna(0)
proc_x[proc_x < 0.09] = 0
proc_x[proc_x > 100000] = 100000

print(len(proc_x.columns))
sort_new = []
for i in proc_x.columns:
  cor = proc_x[i].corr(proc_y)
  if abs(cor) > 0.18:
    print(i, cor)
  if abs(cor) > 0.01:
    sort_new.append((abs(cor), i))
  
final_new = []
pascal = sorted(sort_new, key=lambda x: round(x[0],4), reverse=True)
print(pascal)
print(len(pascal))
for i in sorted(sort_new, key=lambda x: round(x[0],4), reverse=True):
  ok = True
  for j in final_new:
    if proc_x[i[1]].corr(proc_x[j[1]]) > 1.8*i[0]:
      ok = False
  if ok:
    final_new.append(i)
print("Final New", final_new)
print(len(final_new))
cols77 = []
for i in final_new:
  cols77.append(i[1])
"""
selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(proc_x, proc_y)
index_best = selector.get_support(indices=True)

new_columns = map(lambda x: proc_x.columns[x], index_best)
proc_x = proc_x[new_columns]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

proc_x = pd.DataFrame(scaler.fit_transform(proc_x))

print(cols77)
proc_x = df[cols] # .join(proc_x)
proc_x = proc_x.fillna(0)
proc_x[proc_x < 0.09] = 0
proc_x[proc_x > 100000] = 100000

hp = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }


kf = GroupKFold(n_splits=8)
roc_auc_list = []
round_roc_auc = []
for train_index, test_index in kf.split(proc_x, proc_y, groups=group_by):
    X_train, X_test = proc_x.iloc[train_index], proc_x.iloc[test_index]
    y_train, y_test = proc_y.iloc[train_index], proc_y.iloc[test_index] 
    # clf = xgb.XGBClassifier(objective="reg:squarederror", random_state=0, scale_pos_weight=5)
    # clf = RidgeClassifier(class_weight={0: 1, 1: 5})
    clf = LinearRegression()
    clf.fit(X_train, y_train, sample_weight = 2*y_train+1)
    y_pred = clf.predict(X_test)
    y_pred[y_pred < 0] = 0 
    y_pred[y_pred > 1] = 1
    count1=0
    count0=0 
    for i in y_pred[:30]:
      print(round(i,2), end=", ")
      
    print()
    print("Pred Quote", round(sum(np.round(y_pred, 0))/len(y_pred),2), "Train Quote", round(sum(np.round(y_train,0))/len(y_train),2))
    
    roc_auc_list.append(roc_auc_score(y_test, y_pred))
    round_roc_auc.append(roc_auc_score(y_test, np.round(y_pred+0.1,0)))

print(np.mean(roc_auc_list), np.mean(round_roc_auc)) # 0.6886 # 0.73


new_features = ["age_admission_unzmulLACT =lactat", "CRPdivBIC_VB", "LACT =lactatdivKA = Kalium", "BIC_stdivBIC_VB", "LACT =lactatinf7", "GGTmulCR", "frequency_firstdivBIC_st", "NAmulGGT", "GGTinf5", "UREA = HarnstoffdivKA = Kalium", "CRPinf8", "INRiHinf5", "NAdivBIC_VB", "ASATinf5", "UREA = Harnstoff", "frequency_firstdivpH", "GGT", "LACT =lactatinf3", "BIC_st", "pH", "ort_vor_aufnahme_spitalinf2", "temperature_highestinf7", "CR"]



features1 = [to_predict, "pseudoid_patient", "age_admission_unz", "sex", "triage", "referral_unz", "ort_vor_aufnahme_spital", "time_unz", "admission_choice",
                 "pulse_first", "frequency_first", "systolic_bp_first", "systolic_bp_first_inf_100",
                 "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk", "leuk_sup_10", "leuk_inf_4", "leuk_inf_0_5", "entrance_unz_date"]

features2= [to_predict, "pseudoid_patient", "age_admission_unz", "sex", "triage", "referral_unz", "ort_vor_aufnahme_spital", "admission_choice",
                 "pulse_first", "frequency_first", "systolic_bp_first",
                 "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first",
                 "lvl_consc_alert", "spo2_first", "temperature_highest",
                 "o2_gabe", "THZ (Thrombozyten)", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "CRP", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk",]

features3 = [to_predict, "pseudoid_patient", "age_admission_unz", "sex", "respiratory_rate_first", "THZ (Thrombozyten)", "lvl_consc_alert", "o2_gabe", "INRiH", "Hb", "UREA = Harnstoff", "CRP", "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB"]

features4 = [to_predict, "pseudoid_patient", "age_admission_unz", "THZ (Thrombozyten)" , "CRP", "LACT =lactat", "BIC_st", "ASAT"]

features5 = [to_predict, "pseudoid_patient", "age_admission_unz", "CRP", "PCThs"]

features_better = [to_predict, "pseudoid_patient","LACT =lactat", "LACT =lactatinf7", "CRP", "GGT", "GGTinf5",  "CRPinf8",  "INRiHinf5", "INRiH", "THZ (Thrombozyten)", "THZ (Thrombozyten)inf3", "ASAT", "ASATinf3", "UREA = Harnstoff", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "BIC_VB",  "BIC_st",  "BIC_stinf1", "lvl_consc_alert",
                   "pH", "pHinf1", "CR", "CRinf8", "EOS =eosinophile", "EOS =eosinophileinf6", "age_admission_unz", "diastolic_bp_first", "diastolic_bp_firstinf4", "respiratory_rate_first", "respiratory_rate_firstinf3", "systolic_bp_first", "systolic_bp_firstinf4", "Hb", "PCThs", "KA = Kalium", "KA = Kaliuminf2", "pulse_first", "pulse_firstinf5", "NA", "NAinf2",
                   "gcs_inf_15_first"]

features_better2 = [to_predict, "LACT =lactat", "LACT =lactatinf7", "CRP", "GGT", "GGTinf5", "INRiHinf5", "INRiH", "THZ (Thrombozyten)", "THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoff", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "BIC_VB",  "BIC_st",  "BIC_stinf1", "lvl_consc_alert",
                   "pHinf1", "CRinf8", "EOS =eosinophileinf6", "age_admission_unz", "diastolic_bp_firstinf4", "respiratory_rate_firstinf3", "systolic_bp_firstinf4", "Hb", "PCThs"]

features_better3 = [to_predict, "pseudoid_patient", "LACT =lactat", "LACT =lactatinf7", "CRP","GGT", "GGTinf5","INRiHinf5","UREA = Harnstoff","BIC_st", "pH", "lvl_consc_alert", "PCThs"]
from collections import Counter


d =  Counter(features_better)  # -> Counter({4: 3, 6: 2, 3: 1, 2: 1, 5: 1, 7: 1, 8: 1})
res = [k for k, v in d.items() if v > 1]
print(res)


# feature_list = [features1, features2, features3, features4, features5]
# feature_list = [features5]
feature_list = [features_better2]
feature_name = ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"]

import pandas as pd
import numpy as np
df = pd.read_csv("x.csv")
df.entrance_unz_date = pd.to_datetime(df.entrance_unz_date)
df = df.sort_values('entrance_unz_date')

cols = ["age_admission_unz", "sex", "triage", "referral_unz", "ort_vor_aufnahme_spital", "time_unz", "admission_choice",
                 "pulse_first", "frequency_first", "systolic_bp_first", "systolic_bp_first_inf_100",
                 "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first",
                 "lvl_consc_alert", "temperature_highest", "temperature_lowest", "spo2_first",
                 "o2_gabe", "THZ (Thrombozyten)", "NA", "ASAT", "UREA = Harnstoff", "GFR",
                 "PCThs", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "ALAT", "INRiH",
                 "CR", "pH", "LACT =lactat", "GGT", "BIC_st", "BIC_VB", "Leuk", "leuk_sup_10", "leuk_inf_4", "leuk_inf_0_5", "entrance_unz_date"]

y = df['blood_culture_positive']
df = df[cols]

df = df.select_dtypes(['number'])

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_score

pipeline = Pipeline([
    ('imputer', SimpleImputer(missing_values=np.nan, strategy='constant')),
    # ('normalizer', Normalizer()),
    ('model', LinearRegression())
])

scores = cross_val_score(pipeline, df, y, scoring='roc_auc', cv=10)

print(scores)
print(np.mean(scores))

sys.stdout = old_stdout
log_file.close()


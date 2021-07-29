from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import numpy as np

minmaxint = lambda x : int(round(min(1,max(0,x)),0))

cols = ["age_admission_unz", "triage", "referral_unz", "admission_choice", 
"diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first", 
"lvl_consc_alert", "temperature_lowest", "spo2_first", "THZ (Thrombozyten)", 
"NA", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "INRiH", "CR", 
"LACT =lactat", "GGT", "BIC_st", "leuk_sup_10", "leuk_inf_4", "leuk_inf_0_5", 
"LACT =lactatinf7", "GGTinf5", "INRiHinf5", "THZ (Thrombozyten)inf3", 
"ASATinf3", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "pHinf1", 
"EOS =eosinophileinf6","diastolic_bp_firstinf4", "respiratory_rate_firstinf3", 
"systolic_bp_firstinf4", 'Hausarzt (privat)', 'Inselklinik', 
'temperature_lowestinf9', 'EOS =eosinophileinf8', 'frequency_firstinf9', 
'GFRdivEOS =eosinophile', 'EOS =eosinophiledivrespiratory_rate_first', 
'LACT =lactatmultemperature_lowest']

df = pd.read_csv("sepsis_prediction3.csv")

X = df[cols]
y = df["blood_culture_positive"]
group_by = df["pseudoid_patient"]

from sklearn import preprocessing
normalizer = preprocessing.Normalizer()
X = pd.DataFrame(normalizer.fit_transform(X))

kf = GroupKFold(n_splits=8)
roc_auc_list = []
roc_auc_list_int = []
f1_list = []

for train_index, test_index in kf.split(X, y, groups=group_by):
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	y_flatten = y_train.values.flatten()
	bin_count = np.bincount(y_flatten)
	class_weight = bin_count[0]/bin_count[1] - 1
	clf = XGBClassifier()
	clf.fit(X_train, y_flatten, sample_weight=class_weight*y_flatten+1)
	y_pred = clf.predict(X_test)
	y_pred_int = list(map(minmaxint, y_pred))
	roc_auc_list.append(roc_auc_score(y_test, y_pred))
	roc_auc_list_int.append(roc_auc_score(y_test, y_pred_int))
	f1_list.append(f1_score(y_test, y_pred_int))

print("This model:  ", np.mean(roc_auc_list), np.mean(roc_auc_list_int), np.mean(f1_list))
print("Current Best: 0.7350007946670377 0.6980764945460693 0.4397925862599964")

if np.mean(roc_auc_list) > 0.735 and np.mean(roc_auc_list_int) > 0.7 and np.mean(f1_list) > 0.44:
	print("New HighScore")


from sklearn.model_selection import GroupKFold
import autokeras as ak
import tensorflow as tf
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import numpy as np
import kerastuner

def minmaxint(x):
	return int(round(min(1,max(0,x)),0))
    
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


class Metrics(Callback):
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
	 
	def on_epoch_end(self, epoch, logs={}):
		val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
		val_targ = self.model.validation_data[1]
		_val_f1 = f1_score(val_targ, val_predict)

		_val_recall = recall_score(val_targ, val_predict)
		_val_precision = precision_score(val_targ, val_predict)
		self.val_f1s.append(_val_f1)
		self.val_recalls.append(_val_recall)
		self.val_precisions.append(_val_precision)
"""
		print(" — val_f1: {} — val_precision: {} — val_recall {}".format(_val_f1, _val_precision, _val_recall)
"""
metrics = Metrics()

cols = ["age_admission_unz", "triage", "referral_unz", "admission_choice", "diastolic_bp_first", 
"respiratory_rate_first", "gcs_inf_15_first", "lvl_consc_alert", 
"temperature_lowest", "spo2_first", "THZ (Thrombozyten)", "NA", 
"EOS =eosinophile", "CRP", "KA = Kalium", 
"Hb", "INRiH", "CR", "LACT =lactat", "GGT", "BIC_st", "leuk_sup_10", 
"leuk_inf_4", "leuk_inf_0_5", "LACT =lactatinf7", "GGTinf5", "INRiHinf5", 
"THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoffinf3", 
"ort_vor_aufnahme_spitalinf2", "pHinf1", 
"EOS =eosinophileinf6","diastolic_bp_firstinf4", "respiratory_rate_firstinf3", 
"systolic_bp_firstinf4", 'Hausarzt (privat)',
'Inselklinik', 'temperature_lowestinf9', 'EOS =eosinophileinf8', 'frequency_firstinf9', 'GFRdivEOS =eosinophile', 'EOS =eosinophiledivrespiratory_rate_first', 'LACT =lactatmultemperature_lowest']

df = pd.read_csv("sepsis_prediction3.csv")

X = df[cols]
y = df["blood_culture_positive"]
group_by = df["pseudoid_patient"]

kf = GroupKFold(n_splits=8)
roc_auc_list = []
f1_list = []
roc_auc_list_int = []

for train_index, test_index in kf.split(X, y, groups=group_by):
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	bin_count = np.bincount(y_train.values.flatten())
	class_weight = bin_count[0]/bin_count[1]
	clf = ak.StructuredDataClassifier(objective=kerastuner.Objective('val_f1', direction='max'), overwrite=True, max_trials=3)
	clf.fit(X_train, y_train, validation_data=(X_train, y_train), callbacks=[metrics], epochs=10)
	y_pred = clf.predict(X_test)
	y_pred_int = list(map(minmaxint, y_pred))
	roc_auc_list.append(roc_auc_score(y_test, y_pred))
	roc_auc_list_int.append(roc_auc_score(y_test, y_pred_int))
	f1_list.append(f1_score(y_test, y_pred_int))

print("NEW:", np.mean(roc_auc_list), np.mean(roc_auc_list_int), np.mean(f1_list))
print("OLD: 0.7350007946670377 0.6980764945460693 0.4397925862599964")


from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import roc_auc_score, f1_score
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import sklearn
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score
import xgboost as xgb

from sklearn.ensemble import RandomForestClassifier

minmaxint = lambda x : int(round(min(1,max(0,x)),0))

cols = ["age_admission_unz", "triage", "referral_unz", 
"diastolic_bp_first", "gcs_inf_15_first", "lvl_consc_alert", 
"spo2_first", "THZ (Thrombozyten)", "EOS =eosinophile", 
"CRP", "KA = Kalium", "Hb", "LACT =lactat", "GGT", "BIC_st", 
"leuk_sup_10", "leuk_inf_4", "leuk_inf_0_5", 
"LACT =lactatinf7", "INRiHinf5", "THZ (Thrombozyten)inf3", 
"ASATinf3", "UREA = Harnstoffinf3", 
"ort_vor_aufnahme_spitalinf2", "EOS =eosinophileinf6", 
"respiratory_rate_firstinf3", "systolic_bp_firstinf4", 
'Inselklinik', 'temperature_lowestinf9', 'EOS =eosinophileinf8']

df = pd.read_csv("sepsis_prediction5.csv")

X = df[cols]
y = df["blood_culture_positive"]

for i in cols:
	print(i, '\t', df[i].corr(df["blood_culture_positive"]))

# X = pd.DataFrame(sklearn.preprocessing.normalize(X))
print(X)

kf = KFold(n_splits=5)
roc_auc_list = []
roc_auc_list_int = []
f1_list = []
recall_list = []
precision_list = []
accuracy_list = []

"""
tsne = TSNE(n_components=2, verbose=1, random_state=123)
z = tsne.fit_transform(X) 

df = pd.DataFrame()
df["y"] = y
df["comp-1"] = z[:,0]
df["comp-2"] = z[:,1]

plot = sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                palette=sns.color_palette("hls", 2),
                data=df).set(title="T-SNE projection") 

plt.savefig("outputa.png")

"""

from sklearn.linear_model import Lasso
from sklearn.naive_bayes import GaussianNB
from keras.layers import Dense, Dropout, RNN, Bidirectional, LSTM
from fasttrees.fasttrees import FastFrugalTreeClassifier

def NN():
	model = Sequential()
	# model.add(Dense(32, activation='relu', input_dim=30))
	model.add(Bidirectional(LSTM(4, return_sequences=True), input_shape=(1, 30)))
	model.add(Bidirectional(LSTM(2)))
	# model.add(Dense(16, activation='relu'))
	model.add(Dense(1, activation='linear'))
	model.compile(optimizer=Adam(lr=0.01), loss='mse')
	return model


for train_index, test_index in kf.split(X, y):
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	y_flatten = y_train.values.flatten()
	bin_count = np.bincount(y_flatten)
	class_weight = bin_count[0]/bin_count[1] - 1
	
	X_train= np.reshape(X_train.values,(X_train.shape[0], 1, X_train.shape[1]))
	X_test= np.reshape(X_test.values,(X_test.shape[0], 1, X_test.shape[1]))
	
	clf = NN()
	#clf.fit(X_train, y_flatten, epochs=5000, batch_size=len(X_train), sample_weight=class_weight*y_flatten+1)
	
	
	
	# clf = FastFrugalTreeClassifier()
	# clf = LinearRegression()
	# clf = fasttrees
	# print(X_train.values)
	# print(y_train.values.flatten())
	# clf.fit([[1,2],[3,4]],[0,1])
	clf.fit(X_train, y_train.values.flatten() , sample_weight=class_weight*y_flatten+1, epochs=1000, batch_size=len(X_train))
	
	y_pred = clf.predict(X_test)
	y_pred = [i[0] for i in y_pred.tolist()]
	# print(y_pred)
	
	y_pred_int = list(map(minmaxint, y_pred))
	roc_auc_list.append(roc_auc_score(y_test, y_pred))
	roc_auc_list_int.append(roc_auc_score(y_test, y_pred_int))
	f1_list.append(f1_score(y_test, y_pred_int))
	recall_list.append(recall_score(y_test, y_pred_int))
	precision_list.append(precision_score(y_test, y_pred_int))
	accuracy_list.append(accuracy_score(y_test, y_pred_int))

print("This model:  ", np.mean(roc_auc_list), np.mean(roc_auc_list_int), np.mean(f1_list), np.mean(recall_list), np.mean(precision_list), np.mean(accuracy_list))
print("Current Best: 0.7183366311037686 0.6764051644236325 0.45046285449129")

if np.mean(roc_auc_list) > 0.72 and np.mean(roc_auc_list_int) > 0.68 and np.mean(f1_list) > 0.45:
	print("New HighScore")
		

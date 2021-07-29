from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score, f1_score, plot_roc_curve, precision_score, recall_score, accuracy_score, plot_precision_recall_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(0)

"""
In the following the features are defined to train the model. These can be changed as desired to whatever would improve the performance the most.
"""
FEATURE_NAMES = ["age_admission_unz", "triage", "referral_unz", "admission_choice", "diastolic_bp_first", "respiratory_rate_first", "gcs_inf_15_first", "lvl_consc_alert",  "temperature_lowest", "spo2_first", "THZ (Thrombozyten)", "NA", "EOS =eosinophile", "CRP", "KA = Kalium", "Hb", "INRiH", "CR", "LACT =lactat", "GGT", "BIC_st", "leuk_sup_10", "leuk_inf_4", "leuk_inf_0_5", "LACT =lactatinf7", "GGTinf5", "INRiHinf5", "THZ (Thrombozyten)inf3", "ASATinf3", "UREA = Harnstoffinf3", "ort_vor_aufnahme_spitalinf2", "pHinf1", "EOS =eosinophileinf6","diastolic_bp_firstinf4", "respiratory_rate_firstinf3", "systolic_bp_firstinf4", "PLWS", 'Hausarzt (privat)', 'Inselklinik', 'Polizei (In Begleitung)', 'temperature_lowestinf9', 'EOS =eosinophileinf8', 'frequency_firstinf9', 'GFRdivEOS =eosinophile', 'EOS =eosinophiledivrespiratory_rate_first', 'LACT =lactatmultemperature_lowest']
	

def load_data(filename, calculate_mortality=False):
	"""
	load_data reads the CSV file and preprocesses the data.

	:param filename: The filename of the CSV file
	:param calculate_mortality: If it is set to False, the method calculates the blood culture positive, otherwise the mortality in the next 28 days.
	:return: input_data, target_data, pseudoid_patient
	
	We first load the CSV file into a dataframe. This should be a CSV file not an XLSX file as pandas sometimes prefers CSV files. 
	Then the preprocessing:
	1. pd.get_dummies extracts the strings from the specific columns and encodes it into binary. This is used so that those specific strings can be learned by the model.
	2. pd.factorize encodes the object as an enumerated type. This is used so that those specific strings can be learned by the model.
	3. We then go on and delete all non-numeric objects.
	4. For every single column we do a boolean encoding for the 10%, 20%, 30%, up to the 90% percentile. Everything above is encoded as 1 and below as 0. This makes it easier for the model to extract information.
	5. Three feature operations were identified to improve the model, which is just a multiplication or division of a feature pair.
	6. The target value is then defined, either mortality or blood culture positive feature, depending on the function input.
	7. For further simplification of the model we only allow values between 0.09 and 10000.
	"""
	df = pd.read_csv(filename)
	
	df = df.join(pd.get_dummies(df["admission_choice"]))
	df = df.join(pd.get_dummies(df["referral_unz"]))
	
	df.referral_unz = pd.factorize(df.referral_unz)[0]
	df.admission_choice = pd.factorize(df.admission_choice)[0]
	df.ort_vor_aufnahme_spital = pd.factorize(df.ort_vor_aufnahme_spital)[0]
	df = df.select_dtypes(['number'])

	for i in df.columns:
		for j in range(1,10):
			df[i+"inf"+str(j)] = (df[i] >= df[i].quantile(0.1*j)).astype(int) 
		
	df['GFRdivEOS =eosinophile'] = df['GFR']/df['EOS =eosinophile']
	df['EOS =eosinophiledivrespiratory_rate_first'] = df['EOS =eosinophile']/df['respiratory_rate_first']
	df['LACT =lactatmultemperature_lowest'] = df['LACT =lactat']*df['temperature_lowest']

	if calculate_mortality:
		df['mort_28_days'] = df['mort_28_days'].fillna(2)
		df.drop(df[abs(df.mort_28_days) > 1].index, inplace=True)
		proc_x = df[FEATURE_NAMES]
		proc_y = df["mort_28_days"].astype(int)
	else:
		proc_x = df[FEATURE_NAMES]
		proc_y = df["blood_culture_positive"]

	proc_x = proc_x.where(proc_x>0.09, other=0)
	proc_x = proc_x.where(proc_x<10000, other=10000)
	return proc_x, proc_y, df["pseudoid_patient"]

def minmaxint(x):
	"""
	Makes sure that the number is between 0 and 1 and rounded accordingly to an integer.
	"""
	return int(round(min(1,max(0,x)),0))
	
def calculate_class_weight(data):
	"""
	Calculates the weight of the imbalanced dataset.
	"""
	bin_count = np.bincount(data.values.flatten())
	return bin_count[0]/bin_count[1] - 1


kf = GroupKFold(n_splits=8)
roc_auc_list = []
f1_list = []
roc_auc_list_int = []
recall_list = []
precision_list = []
accuracy_list = []

X, y, group_by = load_data("predict_em_masked_blutkulturen.csv", calculate_mortality=False)

"""
kf.split splits the data into n_splits different groups as defined above. Further it needs the group_by, which is the pseudo-id of the patient. This makes sure that the same patient is always in the same set. This split can then be used for cross-validation.
"""
split = list(kf.split(X, y, groups=group_by)) 


for train_index, test_index in split:
	"""
	1. We calculate the X_train, X_test, y_train, y_test sets according to the indices of the for loop.
	2. The class_weights are calculated to make sure the imbalanced dataset is trained accordingly and the positive target values, which are less than the negative values, are still taken into consideration.
	3. Model is defined: In this case LinearRegression, but this can be changed.	
	4. A prediction on the test set is made and evaluated.
	5. All the performance metrics are then saved to a list.
	"""
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	class_weight = calculate_class_weight(y_train)
	
	clf = LinearRegression() # Can be changed to any other model as defined in the thesis
	clf.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)
	
	y_pred = clf.predict(X_test)
	y_pred_int = list(map(minmaxint, y_pred))
	
	roc_auc_list.append(roc_auc_score(y_test, y_pred))
	roc_auc_list_int.append(roc_auc_score(y_test, y_pred_int))
	f1_list.append(f1_score(y_test, y_pred_int))
	recall_list.append(recall_score(y_test, y_pred_int))
	precision_list.append(precision_score(y_test, y_pred_int))
	accuracy_list.append(accuracy_score(y_test, y_pred_int))

"""
Prints the results, which were used in the thesis.
ROC AUC: the receiver operating characteristic score
ROC AUC INT: the same as above but with binary classified first
F1-score: 2 * (precision * recall) / (precision + recall)
Recall: TP/ (TP+FN)
Precision: TP / (TP+FP)
Accuracy: (TP+TN) / (TP+TN+FP+FN)

This file should give the following output:
ROC AUC   ROC AUC INT   F1 -Score   Recall   Precision   Accuracy
0.735     0.698          0.440      0.695    0.323        0.700
"""
print(f"{'ROC AUC':<16}{'ROC AUC INT':<16}{'F1-Score':<16}{'Recall':<16}{'Precision':<16}{'Auccuracy':<16}")	
print("{0:1.3f}\t\t{1:1.3f}\t\t{2:1.3f}\t\t{3:1.3f}\t\t{4:1.3f}\t\t{5:1.3f}".format(np.mean(roc_auc_list), np.mean(roc_auc_list_int), np.mean(f1_list), np.mean(recall_list), np.mean(precision_list), np.mean(accuracy_list)))

"""
The following code is used to print the figures:
"""
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
tprs = []
aucs = []

for i, (train_index, test_index) in enumerate(split):
	"""
	The plot_precision_recall_curve is calculated and the plot_roc_curve for each cross-validation split.
	All the information is saved for the plotting afterwards.
	"""
	X_train, X_test = X.iloc[train_index], X.iloc[test_index]
	y_train, y_test = y.iloc[train_index], y.iloc[test_index]
	class_weight = calculate_class_weight(y_train)
	
	clf2 = RidgeClassifier() # Can be changed to any other classifier-model as defined in the thesis. But it needs be a classifier otherwise it will not work
	clf2.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)
	
	viz = plot_precision_recall_curve(clf2, X_test, y_test, name='Recall/Precision fold {}'.format(i + 1), alpha=0.3, lw=1, ax=ax2)
	viz = plot_roc_curve(clf2, X_test, y_test, name='ROC fold {}'.format(i+1), alpha=0.3, lw=1, ax=ax)  

	interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
	interp_tpr[0] = 0.0
	tprs.append(interp_tpr)
	aucs.append(viz.roc_auc)
	
	
"""
The mean and standard deviation of the curves are calculated.
"""
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
count =np.bincount(y.values.flatten())
positive_rate = count[1]/(count[0]+count[1])


"""
Figure 1: is the ROC Curve of the different folds and then a mean curve over all splits with its deviation is put on top of it.
"""
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC Curve - Positive Blood Culture Prediction")
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
ax.legend(loc="lower right", prop={'size': 7})
fig.savefig('roc_curve_blood_culture_prediction.png', dpi=300)


"""
Figure 2: The Precision / Recall curve of the different folds
"""
ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Precision / Recall - Positive Blood Culture Prediction")
ax2.axhline(positive_rate, color='r', linestyle='--', label='Chance', lw=2, alpha=0.8)
ax2.legend(loc="upper right", prop={'size': 7})
ax2.plot(np.mean(recall_list), np.mean(precision_list), color='r', marker='o', markersize=10)
fig2.savefig('precision_recall_curve_blood_culture_prediction.png', dpi=300)

plt.show()

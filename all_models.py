# 1. Linear Regression
from sklearn.linear_model import LinearRegression
clf.LinearRegression()
clf.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)

# 2. Logistic Regression
from sklearn.linear_model import LogisticRegression
clf.LogisticRegression()
clf.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)

# 3. Ridge Regression
from sklearn.linear_model import Ridge
clf.Ridge()
clf.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)

# 4. Lasso
from sklearn.linear_model import Lasso
clf.Lasso()
clf.fit(X_train, y_train)

# 5. Dense Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(lr=0.01), loss='mse')

model.fit(X_train, y_train.values.flatten() , sample_weight=class_weight*y_train.values.flatten()+1, epochs=1000, batch_size=len(X_train))
	
y_pred = model.predict(X_test)
y_pred = [i[0] for i in y_pred.tolist()]

# 6. XGBoost Regression
import xgboost as xgb
clf = xgb.XGBRegressor(objective ='reg:linear', n_estimators = 10)
clf.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)

# 7. Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)


# 8. MLP Regressor
from sklearn.neural_network import MLPRegressor
clf = MLPRegressor() 
clf.fit(X_train, y_train)


# 9. SVC RBF
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)

# 10. Bidirectional LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam

X_train= np.reshape(X_train.values,(X_train.shape[0], 1, X_train.shape[1]))
X_test= np.reshape(X_test.values,(X_test.shape[0], 1, X_test.shape[1]))
	
model = Sequential()
model.add(Bidirectional(LSTM(4, return_sequences=True), input_shape=(1, 46)))
model.add(Bidirectional(LSTM(2)))
model.add(Dense(1, activation='linear'))
model.compile(optimizer=Adam(lr=0.01), loss='mse')

model.fit(X_train, y_train.values.flatten() , sample_weight=class_weight*y_train.values.flatten()+1, epochs=1000, batch_size=len(X_train))
	
y_pred = model.predict(X_test)
y_pred = [i[0] for i in y_pred.tolist()]

# 11. Random Forest
from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor()
clf.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)

# 12. Decision Tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train, sample_weight=class_weight*y_train.values.flatten()+1)

# 13. Fast and frugal tree
from fasttrees.fasttrees import FastFrugalTreeClassifier
clf = FastFrugalTreeClassifier()
clf.fit(X_train, y_train)

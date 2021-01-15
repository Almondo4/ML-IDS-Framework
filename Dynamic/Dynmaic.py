import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Dynamic_Training.csv")
DataTest = pd.read_pickle("../Data/Dynamic_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

# level 1
from sklearn.linear_model import LogisticRegression
#  Model Building
# import tensorflow as tf
# from tensorflow import keras
# model = keras.models.Sequential()
# model.add(keras.layers.Dense(input_shape = 3, units=6, activation="relu",
#                               name="hiddenL_1"))
# model.add(tf.keras.layers.Dropout(rate =0.15))
# model.add(keras.layers.Dense(units=6, activation="relu", name="hiddenL_3"))
# model.add(tf.keras.layers.Dropout(rate =0.15))
# model.add(keras.layers.Dense(units=1, activation="softmax", name="outLayer"))
#
# # Compiling
# opt = tf.keras.optimizers.Adam()
# model.compile(optimizer=opt, loss ="binary_crossentropy", metrics =["accuracy"])

# Training

# level 0
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
# estimators = [
#     ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
#     ('etc', ExtraTreesClassifier(verbose=1,criterion= 'gini', n_estimators= 300)),
#     ('bg', BaggingClassifier()),
#
# ]
#
# from sklearn.ensemble import StackingClassifier
# clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(verbose=1))
# # Training
# clf = clf.fit(featureMatrixTR,labelVectorTR)
#
# # Testing
# from sklearn.metrics import classification_report,confusion_matrix
# from sklearn.metrics import roc_auc_score
# y_pred2 = clf.predict(featureMatrix)
# print(classification_report(labelVector, y_pred2,digits=4))
# auc = roc_auc_score(labelVector, y_pred2)
# print('ROC AUC: %f' % auc)




# #######################################3

# compare ensemble to each baseline classifier
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier
from matplotlib import pyplot
from sklearn.ensemble import AdaBoostClassifier
# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('rf', RandomForestClassifier(n_estimators=10, random_state=42)))
	# level0.append(('knn', KNeighborsClassifier()))
	# level0.append(('Ada', AdaBoostClassifier(n_estimators=150,algorithm="SAMME.R",)))
	level0.append(('etc', ExtraTreesClassifier(verbose=1,criterion= 'gini', n_estimators= 300)))
	level0.append(('bg', BaggingClassifier()))
	# define meta learner model
	level1 = LogisticRegression(verbose=1)
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model

	# level0.append(('cart', DecisionTreeClassifier()))

# get a list of models to evaluate
def get_models():
    models = dict()
    models['rf'] = RandomForestClassifier(n_estimators=10, random_state=42)
    # models['Kn'] = KNeighborsClassifier()
    # models['Ada'] = AdaBoostClassifier(n_estimators=150,algorithm="SAMME.R",)
    models['etc'] = ExtraTreesClassifier(verbose=1,criterion= 'gini', n_estimators= 300)
    models['bg'] = BaggingClassifier()
    models['stacking'] = get_stacking()
    return models

# evaluate a give model using cross-validation\
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
def evaluate_model(model, X, y):

	print(model)
	cv = model.fit(X,y)
	print(model," Trained")
	y_pred2 = cv.predict(featureMatrix)

	print(classification_report(labelVector, y_pred2,digits=4))

	auc = roc_auc_score(labelVector, y_pred2)
	print('ROC AUC: %f' % auc)
	return cv


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, featureMatrixTR, labelVectorTR)

import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

# ##########################################################################33

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
# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('lr', LogisticRegression()))
	level0.append(('knn', KNeighborsClassifier()))
	level0.append(('cart', DecisionTreeClassifier()))
	level0.append(('svm', SVC()))
	level0.append(('bayes', GaussianNB()))
	# define meta learner model
	level1 = LogisticRegression(verbose=1)
	# define the stacking ensemble
	model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
	return model

# get a list of models to evaluate
def get_models():
	models = dict()
	# models['lr'] = LogisticRegression()
	models['knn'] = KNeighborsClassifier()
	models['cart'] = DecisionTreeClassifier()
	models['svm'] = SVC()
	models['bayes'] = GaussianNB()
	models['stacking'] = get_stacking()
	return models

# evaluate a give model using cross-validation\
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
def evaluate_model(model, X, y):

	print(model)
	cv = model.fit(X,y)
	# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
	print(model," Trained")
	y_pred2 = cv.predict(featureMatrix)

	# print("Performance:",sum(y_pred2==labelVector)/len(labelVector))
	# print("Confusion Matrix:\n",CM(labelVector,y_pred2))



	## TESTING
	# predictions = cb.predict(featureMatrix)

	# XGB_predictions_Classes =model.predict_classes(test)
	#
	# cm = confusion_matrix(labelVector, y_pred2)
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
	# results.append(scores)
	# names.append(name)
	# print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# # plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()

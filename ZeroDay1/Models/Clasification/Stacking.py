import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../Data/AndMal_Zero_Train.csv")
DataTest = pd.read_pickle("../../Data/AndMal_Zero_Test.csv")
DataZero = pd.read_pickle("../../Data/AndMal_Zero_Day.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values
featureMatrixZ = DataZero.iloc[:,:-1].values
labelVectorZ = DataZero.iloc[:,-1].values


#       3 Scaling the dataSet
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
featureMatrixTR = sc.fit_transform(featureMatrixTR)
featureMatrix = sc.fit_transform(featureMatrix)
featureMatrixZ = sc.fit_transform(featureMatrixZ)


# Training

# level 0
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier




# #######################################3

# compare ensemble to each baseline classifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import StackingClassifier

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
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score

def evaluate_model(model, X, y,labelVector):

	print(model)
	cv = model.fit(X,y)
	print(model," Trained")
	RF_predictions = model.predict(featureMatrix)
	print(RF_predictions)
	print(classification_report(labelVector, RF_predictions,digits=4))

	auc = roc_auc_score(labelVector, RF_predictions,multi_class="ovo")
	print('ROC AUC: %f' % auc)
	print("############################################################ ZERO DAY")
	RF_predictions = model.predict(featureMatrixZ)
	print(classification_report(labelVectorZ, RF_predictions,digits=4))
	auc = roc_auc_score(labelVectorZ,RF_predictions)
	print('ROC AUC: %f' % auc)



# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, featureMatrixTR, labelVectorTR,labelVector)

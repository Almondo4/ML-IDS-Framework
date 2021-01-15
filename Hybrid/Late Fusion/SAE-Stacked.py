
# DATA
# # Dynamic
import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../../Data/Dynamic_Training.csv")
DataTest = pd.read_pickle("../../Data/Dynamic_Testing.csv")

featureMatrixTRD = DataTrain.iloc[:,:-1].values
labelVectorTRD = DataTrain.iloc[:,-1].values
featureMatrixD = DataTest.iloc[:,:-1].values
labelVectorD = DataTest.iloc[:,-1].values

# # Static

DataTrain = pd.read_pickle("../../Data/Static_Training.csv")
DataTest = pd.read_pickle("../../Data/Static_Testing.csv")

featureMatrixTRS = DataTrain.iloc[:,:-1].values
labelVectorTRS = DataTrain.iloc[:,-1].values
featureMatrixS = DataTest.iloc[:,:-1].values
labelVectorS = DataTest.iloc[:,-1].values
# # Hybrid
DataTrain = pd.read_pickle("../../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../../Data/Hybrid_Testing.csv")

featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)

# Models
from tensorflow import keras


staticModel = keras.models.load_model("../Late Integration/SAE_DLS_2.h5")
dynamicModel = keras.models.load_model("../Late Integration/SAE_DLD_2.h5")

# Creating Datasets
# # Training
staticSet=staticModel.predict(featureMatrixTRS)
dynamicSet=dynamicModel.predict(featureMatrixTRD)
DATATRaining = np.concatenate((staticSet,dynamicSet), axis =1)
# # Testing
staticSet=staticModel.predict(featureMatrixS)
dynamicSet=dynamicModel.predict(featureMatrixD)
DATATesting = np.concatenate((staticSet,dynamicSet), axis =1)


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
def evaluate_model(model, X, y):

	print(model)
	cv = model.fit(X,y)
	print(model," Trained")
	y_pred2 = cv.predict(DATATesting)

	print(classification_report(labelVector, y_pred2,digits=4))

	auc = roc_auc_score(labelVector, y_pred2)
	print('ROC AUC: %f' % auc)
	return cv


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, DATATRaining, labelVectorTR)

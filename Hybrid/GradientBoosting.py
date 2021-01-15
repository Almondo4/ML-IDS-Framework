import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("../Data/Hybrid_Training.csv")
DataTest = pd.read_pickle("../Data/Hybrid_Testing.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


# #       3 Scaling the dataSet
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# featureMatrixTR = sc.fit_transform(featureMatrixTR)
# featureMatrix = sc.fit_transform(featureMatrix)

# from tensorflow.keras.utils import to_categorical
# labelVector = to_categorical(labelVector)
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
labelVectorTR = labelencoder.fit_transform(labelVectorTR)
labelVector = labelencoder.fit_transform(labelVector)


# # Feature Extraction
# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif
#
# selector = SelectKBest(f_classif, k=100)
# selected_features = selector.fit_transform(featureMatrix, labelVector)
#
# print((-selector.scores_).argsort()[:])


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix as CM
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import roc_auc_score
lr_list = [0.5]
# 0.05, 0.075, 0.1, 0.25, 0.5, 0.75,
for learning_rate in lr_list:
    gb_clf = GradientBoostingClassifier(n_estimators=10, learning_rate=learning_rate,
                                        max_features=20, max_depth=420, random_state=0,verbose=1)
    gb_clf.fit(featureMatrixTR,labelVectorTR)
    y_pred2=gb_clf.predict(featureMatrix)
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb_clf.score(featureMatrixTR,labelVectorTR)))
    print("Accuracy score (validation): {0:.3f}".format(gb_clf.score(featureMatrix,labelVector)))
    cm = confusion_matrix(labelVector, y_pred2)
    print(classification_report(labelVector, y_pred2))
    auc = roc_auc_score(labelVector, y_pred2)
    print('ROC AUC: %f' % auc)




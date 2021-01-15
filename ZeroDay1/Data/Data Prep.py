import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("AndMal_2020_Train.csv")
DataTest = pd.read_pickle("AndMal_2020_Test.csv")
DataZero = pd.read_pickle("ZD.csv")
Benign = pd.read_csv("Ben0.csv")

Benign["Class"]= "Benign"
Benign = Benign.iloc[3000:3500,1:].values

DataZero= np.concatenate(( Benign,DataZero))
DataZero= pd.DataFrame(data=DataZero)
DataZero = DataZero.sample(frac=1).reset_index(drop=True)

featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values
featureMatrixZ = DataZero.iloc[:,:-1].values
labelVectorZ = DataZero.iloc[:,-1].values

lbx1=np.array([])
for x in labelVectorTR:
    if x == "Benign":
        lbx1 = np.append (lbx1, ["0"])
    else:
        lbx1 = np.append (lbx1, ["1"])

lbx2=np.array([])
for x in labelVector:
    if x == "Benign":
        lbx2 = np.append (lbx2, ["0"])
    else:
        lbx2 = np.append (lbx2, ["1"])

lbx3=np.array([])
for x in labelVectorZ:
    if x == "Benign":
        lbx3 = np.append (lbx3, ["0"])
    else:
        lbx3 = np.append (lbx3, ["1"])



DataTrain = np.append(featureMatrixTR, lbx1[:, None], axis=1)
DataTrain = pd.DataFrame(data=DataTrain)
DataTrain.to_pickle(path="AndMal_Zero_Train.csv")
DataTest = np.append(featureMatrix, lbx2[:, None], axis=1)
DataTest = pd.DataFrame(data=DataTest)
DataTest.to_pickle(path="AndMal_Zero_Test.csv")
DataZero =np.append(featureMatrixZ, lbx3[:, None], axis=1)
DataZero = pd.DataFrame(data=DataZero)
DataZero.to_pickle(path="AndMal_Zero_Day.csv")




import pandas as pd
import numpy as np
DataTrain = pd.read_pickle("AndMal_2020_NoZD_Train.csv")
DataTest = pd.read_pickle("AndMal_2020_NoZD_Test.csv")


featureMatrixTR = DataTrain.iloc[:,:-1].values
labelVectorTR = DataTrain.iloc[:,-1].values
featureMatrix = DataTest.iloc[:,:-1].values
labelVector = DataTest.iloc[:,-1].values


DataTrain = pd.read_pickle("./2/AndMal_2020_Train.csv")
DataTest = pd.read_pickle("./2/AndMal_2020_Test.csv")

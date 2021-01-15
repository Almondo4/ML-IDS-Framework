

Aller au contenu
Utiliser Gmail avec un lecteur d'écran

Recherche
Rechercher dans les messages


Meet
Nouvelle réunion
Rejoindre une réunion
Hangouts

9 sur 3 261
yours
Boîte de réception

mohamed batouche
Pièces jointes
7 nov. 2020 18:47 (il y a 2 jours)
À moi


Traduire le message
Désactiver pour : malais
salam
Zone contenant les pièces jointes

# This is a Python script for Breast Cancer Subtyping based on deep learning and dynamic clustering.
# 2 Stacked Auto-Encoder + 1 Auto-encoder are used first for dimensionality reduction
# Then Elbow clustering is used to help selecting the best number of clusters ...
# This corresponds to the second model using two Stacked Auto-Encoder [421 -> 200 -> 50 -> 200 -> 421] with mRNA data and expression data
# and an Auto-Encoder [50+50 -> 50 -> 100]
# dataset: mRNA data (Stacked Matrix) - Early Integration ...


from pandas import read_csv
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# use mRNA data to generate codings1 matrix ...

Dataset = read_csv("../../Data/our_data_mRNAnew.csv")
A = Dataset.iloc[:,2:].values

# 36 rows x 421 columns
print(A)

nbfeatures = len(A[1,:])

print(nbfeatures)

#scale data between [0..1]
scaler = MinMaxScaler()
scaler.fit(A)
X_scaled = scaler.transform(A)
print(X_scaled)

# use Stacked auto-encoder for dimensionality reduction: nbfeatures:421 -> 200 -> 50
encoder1 = keras.models.Sequential([keras.layers.Dense(200, input_shape=[nbfeatures]),
                                   keras.layers.Dense(50, input_shape=[200])])

decoder1 = keras.models.Sequential([keras.layers.Dense(200, input_shape=[50]),
                                   keras.layers.Dense(nbfeatures, input_shape=[200])])

sae1 = keras.models.Sequential([encoder1, decoder1])
sae1.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.0001))

X_train = X_scaled

history = sae1.fit(X_train, X_train, epochs=25000)
codings1 = encoder1.predict(X_train)
# print(codings1)

# use expression data to generate codings2 matrix ...

Dataset = read_csv("../../Data/our_data_expressionnew.csv")
B = Dataset.iloc[:,2:].values

# 36 rows x 421 columns
print(B)

nbfeatures = len(B[1,:])

print(nbfeatures)

#scale data between [0..1]
scaler = MinMaxScaler()
scaler.fit(B)
X_scaled = scaler.transform(B)
print(X_scaled)

# use Stacked auto-encoder for dimensionality reduction: nbfeatures:421 -> 200 -> 50
encoder2 = keras.models.Sequential([keras.layers.Dense(200, input_shape=[nbfeatures]),
                                   keras.layers.Dense(50, input_shape=[200])])

decoder2 = keras.models.Sequential([keras.layers.Dense(200, input_shape=[50]),
                                   keras.layers.Dense(nbfeatures, input_shape=[200])])

sae2 = keras.models.Sequential([encoder2, decoder2])
sae2.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.0001))

X_train = X_scaled

history = sae2.fit(X_train, X_train, epochs=25000)
codings2 = encoder2.predict(X_train)
# print(codings2)

# use another Auto-Encoder for dimensionality reduction ...

C = np.concatenate((codings1, codings2),axis=1)

print("")

print("codings1 ")
print(codings1)
print("codings2 ")
print(codings2)
print("concatenate")
print(C)

print("")
print(len(codings1[1,:]))
print(len(codings2[1,:]))
print(len(C[1,:]))

nbfeatures = len(C[1,:])

encoder3 = keras.models.Sequential([keras.layers.Dense(50, input_shape=[nbfeatures])])

decoder3 = keras.models.Sequential([keras.layers.Dense(nbfeatures, input_shape=[50])])

ae = keras.models.Sequential([encoder3, decoder3])
ae.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=0.0001))

X_train = C

history = ae.fit(X_train, X_train, epochs=10000)
codings3 = encoder3.predict(X_train)

# use Elbow Clustering to determine the best number of clusters ...
myData = codings3

distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k, init='k-means++')
    kmeanModel.fit(myData)
    distortions.append(kmeanModel.inertia_)

print(distortions)

# display Elbow Method results and select the best k
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()

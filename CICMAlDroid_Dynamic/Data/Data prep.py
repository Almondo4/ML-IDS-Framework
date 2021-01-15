import pandas as pd
import numpy as np
#
# Dataset = pd.read_csv("feature_vectors_syscallsbinders_frequency_5_Cat.csv")
# Dataset = Dataset.sample(frac=1).reset_index(drop=True)
# AndMal_2020_Train = Dataset.iloc[:9000,:]
# AndMal_2020_Train.to_pickle("CICMalDroid_Train.csv")
# AndMal_2020_Test = Dataset.iloc[9001:,:]
# AndMal_2020_Test.to_pickle("CICMalDroid_Test.csv")


from collections import Counter
Dataset = pd.read_pickle("CICMalDroid_Train.csv")
print("Number of Dataset Samples: ",Dataset.shape[0],", including ",Dataset.shape[1]," Features.")

target = Dataset.values[:,-1]
counter = Counter(target)
D={}
for k,v in counter.items():
	per = v / len(target) * 100
	print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))


for k,v in counter.items():
    per = v / len(target) * 100
    D[k]=per



from matplotlib import pyplot as plt



# Dataset.hist(x, 50, normed=1, facecolor='green', alpha=0.75))
# # show the plot
plt.xlabel('Category')
plt.ylabel('Percentage')

plt.ylim(0,100)

plt.bar(range(len(D)), list(D.values()),)
plt.xticks(range(len(D)), list(D.keys()))

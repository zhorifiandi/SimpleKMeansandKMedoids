from kmeans import KMeans
import utils
import pickle
import numpy as np
from sklearn.metrics import adjusted_rand_score

data = [
		(2,10)
		,(2,5)
		,(8,4)
		,(5,8)
		,(7,5)
		,(6,4)
		,(1,2)
		,(4,9)
		]

dataset = utils.create_list_dataset("CencusIncome.data.txt")
list_label = utils.create_list_label("CencusIncome.data.txt")
# list_label = [0, 2, 1, 0, 1, 1, 2, 0]
# for data in dataset:
# 	print(len(data), data)

classifier = KMeans(n_clusters= 2)

classifier.train(input_data=dataset,normalize="normalize")

list_clustered = classifier.label_clustered
print("ARI SCORE: " + str(adjusted_rand_score(np.array(list_label), np.array(list_clustered))))
print("Model:")
print(classifier.centroids)
pickle.dump(classifier, file=open("kmeans_model.cil","wb"))



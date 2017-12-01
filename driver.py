from kmeans import KMeans
import utils
import pickle

data = [
		(2,10,10)
		,(2,5,7)
		,(8,4,8)
		,(5,8,8)
		,(7,5,10)
		,(6,4,10)
		,(1,2,3)
		,(4,9,10)
		]

dataset = utils.create_list_dataset("CencusIncome.data.txt")
# for data in dataset:
# 	print(len(data), data)

classifier = KMeans(n_clusters= 3)

classifier.train(input_data=dataset)

print("Model:")
print(classifier.centroids)
pickle.dump(classifier, file=open("kmeans_model.cil","wb"))



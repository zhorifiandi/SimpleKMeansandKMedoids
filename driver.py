from kmeans import KMeans
import utils
import pickle
import numpy as np
import json

dataset = utils.create_list_dataset("CencusIncome.data.txt")
datatest = utils.create_list_dataset("CencusIncome.test.txt")
dataset_normalized = utils.normalize_attr(dataset)
datatest_normalized = utils.normalize_attr(datatest)

# K-MEANS
classifier = KMeans(n_clusters= 2)
classifier.train(input_data=dataset,normalize="normalize")

list_label = utils.create_list_label("CencusIncome.data.txt")
list_label_test = utils.create_list_label("CencusIncome.test.txt")
classifier.full_validation(list_label)
classifier.test_validation(datatest, list_label_test)

# K-MEANS SK-LEARN
utils.test_using_sklearn(list_label,list_label_test,dataset_normalized,datatest_normalized)

pickle.dump(classifier, file=open("kmeans_model.pkl","wb"))

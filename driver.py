from kmeans import KMeans
import utils
import pickle
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score
import json

dataset = utils.create_list_dataset("CencusIncome.data.txt")
list_label = utils.create_list_label("CencusIncome.data.txt")
classifier = KMeans(n_clusters= 2)

classifier.train(input_data=dataset,normalize="normalize")
datatest = utils.normalize_attr(utils.create_list_dataset("CencusIncome.test.txt"))
list_label_test = utils.create_list_label("CencusIncome.test.txt")
list_clustered_test = []
for instance in datatest:
	list_clustered_test.append(classifier.predict(instance))

dataset_normalized = utils.normalize_attr(dataset)
list_label_train_sklearn, list_label_test_sklearn = utils.test_using_sklearn(dataset_normalized, datatest)


# Evaluation for Full Training
print("------------------------ K-MEANS MODEL --------------------------------")
list_clustered = classifier.label_clustered
print("ARI SCORE: " + str(adjusted_rand_score(np.array(list_label), np.array(list_clustered))))
print("MUTUAL INFO SCORE: " + str(adjusted_mutual_info_score(np.array(list_label), np.array(list_clustered))))
print("HOMOGENEITY SCORE: " + str(homogeneity_score(np.array(list_label), np.array(list_clustered))))
print("COMPLETENESS SCORE: " + str(completeness_score(np.array(list_label), np.array(list_clustered))))
print("V MEASURE SCORE: " + str(v_measure_score(np.array(list_label), np.array(list_clustered))))
# np.seterr(all='warn')
# np.multiply.reduce(np.arange(21)+1)
print("FOWLKES-MALLOWS SCORE: " + str(fowlkes_mallows_score(np.array(list_label), np.array(list_clustered))))
# print("SILHOUETTE SCORE: " + str(silhouette_score(np.array(dataset), np.array(list_label), metric="euclidean")))
print("CALINSKI-HARABAZ SCORE: " + str(calinski_harabaz_score(np.array(dataset_normalized), np.array(list_label))))

# Evaluation for Split Validation
print("---------------- K-MEANS MODEL USING DATA TEST ------------------------")
print("ARI SCORE: " + str(adjusted_rand_score(np.array(list_label_test), np.array(list_clustered_test))))
print("MUTUAL INFO SCORE: " + str(adjusted_mutual_info_score(np.array(list_label_test), np.array(list_clustered_test))))
print("HOMOGENEITY SCORE: " + str(homogeneity_score(np.array(list_label_test), np.array(list_clustered_test))))
print("COMPLETENESS SCORE: " + str(completeness_score(np.array(list_label_test), np.array(list_clustered_test))))
print("V MEASURE SCORE: " + str(v_measure_score(np.array(list_label_test), np.array(list_clustered_test))))
# np.seterr(all='warn')
# np.multiply.reduce(np.arange(21)+1)
print("FOWLKES-MALLOWS SCORE: " + str(fowlkes_mallows_score(np.array(list_label_test), np.array(list_clustered_test))))
# print("SILHOUETTE SCORE: " + str(silhouette_score(np.array(dataset), np.array(list_label), metric="euclidean")))
print("CALINSKI-HARABAZ SCORE: " + str(calinski_harabaz_score(np.array(datatest), np.array(list_label_test))))


# Evaluation for Full Training
print("------------------------ SCIKIT LEARN --------------------------------")
print("------------------------ K-MEANS MODEL --------------------------------")
print("ARI SCORE: " + str(adjusted_rand_score(np.array(list_label), np.array(list_label_train_sklearn))))
print("MUTUAL INFO SCORE: " + str(adjusted_mutual_info_score(np.array(list_label), np.array(list_label_train_sklearn))))
print("HOMOGENEITY SCORE: " + str(homogeneity_score(np.array(list_label), np.array(list_label_train_sklearn))))
print("COMPLETENESS SCORE: " + str(completeness_score(np.array(list_label), np.array(list_label_train_sklearn))))
print("V MEASURE SCORE: " + str(v_measure_score(np.array(list_label), np.array(list_label_train_sklearn))))
# np.seterr(all='warn')
# np.multiply.reduce(np.arange(21)+1)
print("FOWLKES-MALLOWS SCORE: " + str(fowlkes_mallows_score(np.array(list_label), np.array(list_label_train_sklearn))))
# print("SILHOUETTE SCORE: " + str(silhouette_score(np.array(dataset), np.array(list_label), metric="euclidean")))
print("CALINSKI-HARABAZ SCORE: " + str(calinski_harabaz_score(np.array(dataset_normalized), np.array(list_label))))

# Evaluation for Split Validation
print("---------------- K-MEANS MODEL USING DATA TEST ------------------------")
print("ARI SCORE: " + str(adjusted_rand_score(np.array(list_label_test), np.array(list_label_test_sklearn))))
print("MUTUAL INFO SCORE: " + str(adjusted_mutual_info_score(np.array(list_label_test), np.array(list_label_test_sklearn))))
print("HOMOGENEITY SCORE: " + str(homogeneity_score(np.array(list_label_test), np.array(list_label_test_sklearn))))
print("COMPLETENESS SCORE: " + str(completeness_score(np.array(list_label_test), np.array(list_label_test_sklearn))))
print("V MEASURE SCORE: " + str(v_measure_score(np.array(list_label_test), np.array(list_label_test_sklearn))))
# np.seterr(all='warn')
# np.multiply.reduce(np.arange(21)+1)
print("FOWLKES-MALLOWS SCORE: " + str(fowlkes_mallows_score(np.array(list_label_test), np.array(list_label_test_sklearn))))
# print("SILHOUETTE SCORE: " + str(silhouette_score(np.array(dataset), np.array(list_label), metric="euclidean")))
print("CALINSKI-HARABAZ SCORE: " + str(calinski_harabaz_score(np.array(datatest), np.array(list_label_test))))



pickle.dump(classifier, file=open("kmeans_model.pkl","wb"))

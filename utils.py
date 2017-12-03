import numpy
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score


# Input must be tuple of integer / float
def countDistance(centroid, instance):
	numpy_centroid = numpy.array(centroid)
	numpy_instance = numpy.array(instance)
	return numpy.linalg.norm(numpy_centroid-numpy_instance)

def create_list_dataset(filename):
	file = open(filename, 'r')
	list_output = []
	for line in file:
		list_temp = []
		for attr in line.split(",")[:-1]:
			attr = attr.strip()
			if attr.isdigit():
				list_temp.append(int(attr))
		if len(list_temp) > 0:
			list_output.append(list_temp)
	list_output = list_output
	return list_output

def normalize_attr(list_dataset):
	array_to_normalize = numpy.array(list_dataset)
	array_to_normalize = normalize(array_to_normalize)
	list_intance = []
	for instance in array_to_normalize:
		list_attr = []
		for attr in instance:
			list_attr.append(attr)
		list_intance.append(list_attr)
	return list_intance

def create_list_label(filename):
	file = open(filename, 'r')
	list_dataset = []
	for line in file:
		if "," in line:
			list_dataset.append(line.split(","))
	# Asumption label value located at last element of list
	
	# Create Map
	b = 0
	dict_label = {}
	for instance in list_dataset:
		if str(instance[-1]) not in dict_label:
			dict_label[str(instance[-1])] = b
			b += 1

	# Create list label
	list_label = []
	for instance in list_dataset:
		list_label.append(dict_label[str(instance[-1])])

	return list_label

def test_using_sklearn(label_true,label_true_test,dataset,datatest):
	X = numpy.array(dataset)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
	cluster_train = kmeans.labels_
	arr_test = numpy.array(datatest)
	cluster_test = kmeans.predict(arr_test)
	

	# Evaluation for Full Training
	print("\n------------------------ SCIKIT LEARN --------------------------------")
	print("--------------- K-MEANS SCORE USING DATA TRAIN -----------------------")
	print("ARI SCORE: " + str(adjusted_rand_score(numpy.array(label_true), numpy.array(cluster_train))))
	print("MUTUAL INFO SCORE: " + str(adjusted_mutual_info_score(numpy.array(label_true), numpy.array(cluster_train))))
	print("HOMOGENEITY SCORE: " + str(homogeneity_score(numpy.array(label_true), numpy.array(cluster_train))))
	print("COMPLETENESS SCORE: " + str(completeness_score(numpy.array(label_true), numpy.array(cluster_train))))
	print("V MEASURE SCORE: " + str(v_measure_score(numpy.array(label_true), numpy.array(cluster_train))))
	print("FOWLKES-MALLOWS SCORE: " + str(fowlkes_mallows_score(numpy.array(label_true), numpy.array(cluster_train))))
	# print("SILHOUETTE SCORE: " + str(silhouette_score(numpy.array(dataset), numpy.array(label_true), metric="euclidean")))
	print("CALINSKI-HARABAZ SCORE: " + str(calinski_harabaz_score(numpy.array(dataset), numpy.array(label_true))))

	# Evaluation for Split Validation
	print("--------------- K-MEANS SCORE USING DATA TEST -----------------------")
	print("ARI SCORE: " + str(adjusted_rand_score(numpy.array(label_true_test), numpy.array(cluster_test))))
	print("MUTUAL INFO SCORE: " + str(adjusted_mutual_info_score(numpy.array(label_true_test), numpy.array(cluster_test))))
	print("HOMOGENEITY SCORE: " + str(homogeneity_score(numpy.array(label_true_test), numpy.array(cluster_test))))
	print("COMPLETENESS SCORE: " + str(completeness_score(numpy.array(label_true_test), numpy.array(cluster_test))))
	print("V MEASURE SCORE: " + str(v_measure_score(numpy.array(label_true_test), numpy.array(cluster_test))))
	print("FOWLKES-MALLOWS SCORE: " + str(fowlkes_mallows_score(numpy.array(label_true_test), numpy.array(cluster_test))))
	# print("SILHOUETTE SCORE: " + str(silhouette_score(numpy.array(dataset), numpy.array(label_true_test), metric="euclidean")))
	print("CALINSKI-HARABAZ SCORE: " + str(calinski_harabaz_score(numpy.array(datatest), numpy.array(label_true_test))))

	return None
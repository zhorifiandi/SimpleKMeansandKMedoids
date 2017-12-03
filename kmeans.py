import utils
import random as rand
import numpy
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import completeness_score
from sklearn.metrics import v_measure_score
from sklearn.metrics import fowlkes_mallows_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

class KMeans:
	# Number of Clusters
	n_clusters = 5

	# List of Centroids (Tuple)
	# This is the model
	centroids = []

	# List All Cluster Membership
	cluster_membership = []

	# Labels
	label_clustered = []

	# Dataset 
	dataset = []


	def __init__(self, n_clusters=5):
		self.n_clusters = n_clusters

	def train(self,input_data,normalize="none"):
		# Normalization (optional)
		if normalize=="normalize":
			input_data = utils.normalize_attr(input_data)

		dataset = input_data

		# Insert input datasets
		temp_input_data = []
		for data in input_data:
			temp_input_data.append(data)

		# Initialize list of clusters
		list_of_clusters = []	
		for i in range(0,self.n_clusters):
			list_of_clusters.append([])

		# Initialize label clustered
		temp_label_clustered = []

		# Choose random centroids for initialization
		current_centroids = []
		for i in range(0,self.n_clusters):
			elmt = rand.choice(temp_input_data)
			temp_input_data.remove(elmt)
			current_centroids.append(elmt)
			list_of_clusters[i].append(elmt)

		instance_change = -1

		
		iteration = 1
		while (instance_change != 0) :
			print("Epoch",iteration)
			if instance_change != -1:
				# Initialize list of clusters			
				list_of_clusters = []
				for i in range(0,self.n_clusters):
					list_of_clusters.append([])

			# Assign each instance to cluster 
			for instance in input_data:
				# Count distance of instance to cluster
				distances_to_cluster = []
				for ctr in current_centroids:
					distances_to_cluster.append(utils.countDistance(ctr,instance))

				# Assign the instance to "nearest" centroid cluster
				min_index = distances_to_cluster.index(min(distances_to_cluster))
				list_of_clusters[min_index].append(instance)

			# Update centroids
			for i,cluster in enumerate(list_of_clusters):
				distances = []
				for data in cluster:
					distances.append(utils.countDistance(current_centroids[i],data))

				current_centroids[i] = tuple(map(lambda y: sum(y) / float(len(y)), zip(*cluster)))

			# Check for any instance change
			if instance_change == -1:
				# If first epoch, skip it
				self.cluster_membership = list_of_clusters
				instance_change = 1
			else:
				instance_change = 0
				for i,clst in enumerate(list_of_clusters):
					list_of_clusters[i].sort()

				for i,clst in enumerate(list_of_clusters):
					self.cluster_membership[i].sort()

				if (self.cluster_membership != list_of_clusters):
					instance_change = 1

				self.cluster_membership = list_of_clusters
			
			print("Result: ")
			print(current_centroids)

			for i,cluster in enumerate(self.cluster_membership):
				print("Total Member of Cluster",i," :",len(cluster))
			iteration += 1

		self.centroids = current_centroids
		print("\n\nEnd Of Epoch")
		print("Result: ")
		for i, ctd in enumerate(self.centroids):
			print("Centroid",i,":",ctd)

		for i,cluster in enumerate(self.cluster_membership):
			print("Total Member of Cluster",i," :",len(cluster))
		iteration += 1
		
		# Converged, create map
		dict = {}
		idx_cluster = 0
		for cluster in self.cluster_membership:
			for instance in cluster:
				dict[str(instance)] = idx_cluster
			idx_cluster +=1

		# Create label
		for instance in input_data:
			temp_label_clustered.append(dict[str(instance)])

		self.label_clustered = temp_label_clustered

	def predict(self, instance):
		distances_to_cluster = []
		for ctr in self.centroids:
			distances_to_cluster.append(utils.countDistance(ctr,instance))

		# Assign the instance to "nearest" centroid cluster
		min_index = distances_to_cluster.index(min(distances_to_cluster))
		return min_index

	def full_validation(self, label_true):
		print("---------------- K-MEANS SCORE USING DATA TRAIN ------------------------")
		print("ARI SCORE: " + str(adjusted_rand_score(numpy.array(label_true), numpy.array(self.label_clustered))))
		print("MUTUAL INFO SCORE: " + str(adjusted_mutual_info_score(numpy.array(label_true), numpy.array(self.label_clustered))))
		print("HOMOGENEITY SCORE: " + str(homogeneity_score(numpy.array(label_true), numpy.array(self.label_clustered))))
		print("COMPLETENESS SCORE: " + str(completeness_score(numpy.array(label_true), numpy.array(self.label_clustered))))
		print("V MEASURE SCORE: " + str(v_measure_score(numpy.array(label_true), numpy.array(self.label_clustered))))
		print("FOWLKES-MALLOWS SCORE: " + str(fowlkes_mallows_score(numpy.array(label_true), numpy.array(self.label_clustered))))
		# print("SILHOUETTE SCORE: " + str(silhouette_score(numpy.array(dataset), np.array(label_true), metric="euclidean")))
		# print("CALINSKI-HARABAZ SCORE: " + str(calinski_harabaz_score(numpy.array(self.dataset), numpy.array(label_true))))

	def test_validation(self, datatest, label_true):
		label_clustered_test = []
		datatest = utils.normalize_attr(datatest)
		for instance in datatest:
			label_clustered_test.append(self.predict(instance))

		print("---------------- K-MEANS SCORE USING DATA TEST ------------------------")
		print("ARI SCORE: " + str(adjusted_rand_score(numpy.array(label_true), numpy.array(label_clustered_test))))
		print("MUTUAL INFO SCORE: " + str(adjusted_mutual_info_score(numpy.array(label_true), numpy.array(label_clustered_test))))
		print("HOMOGENEITY SCORE: " + str(homogeneity_score(numpy.array(label_true), numpy.array(label_clustered_test))))
		print("COMPLETENESS SCORE: " + str(completeness_score(numpy.array(label_true), numpy.array(label_clustered_test))))
		print("V MEASURE SCORE: " + str(v_measure_score(numpy.array(label_true), numpy.array(label_clustered_test))))
		print("FOWLKES-MALLOWS SCORE: " + str(fowlkes_mallows_score(numpy.array(label_true), numpy.array(label_clustered_test))))
		# print("SILHOUETTE SCORE: " + str(silhouette_score(numpy.array(datatest), np.array(label_true), metric="euclidean")))
		print("CALINSKI-HARABAZ SCORE: " + str(calinski_harabaz_score(numpy.array(datatest), numpy.array(label_true))))



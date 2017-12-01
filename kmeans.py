import utils
import random as rand
import numpy

class KMeans:
	# Number of Clusters
	n_clusters = 5

	# List of Centroids (Tuple)
	# This is the model
	centroids = []

	# List All Cluster Membership
	cluster_membership = []

	#WIEGA
	label_clustered = []


	def __init__(self, n_clusters):
		self.n_clusters = n_clusters

	def train(self,input_data,normalize="none"):
		# Normalization (optional)
		if normalize=="normalize":
			input_data = utils.normalize_attr(input_data)

		# Insert input datasets
		temp_input_data = []
		for data in input_data:
			temp_input_data.append(data)
		print(temp_input_data[0])

		# Initialize list of clusters
		list_of_clusters = []	
		for i in range(0,self.n_clusters):
			list_of_clusters.append([])

		# Choose random centroids for initialization
		for i in range(0,self.n_clusters):
			elmt = rand.choice(temp_input_data)
			temp_input_data.remove(elmt)
			self.centroids.append(elmt)
			list_of_clusters[i].append(elmt)

		# print("init")
		# print(list_of_clusters)

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
			# print("Assign each instance to cluster ")
			# print(input_data)
			for instance in input_data:
				# Count distance of instance to cluster
				distances_to_cluster = []
				for ctr in self.centroids:
					distances_to_cluster.append(utils.countDistance(ctr,instance))

				# Assign the instance to "nearest" centroid cluster
				min_index = distances_to_cluster.index(min(distances_to_cluster))
				list_of_clusters[min_index].append(instance)

			# Update centroids
			for i,cluster in enumerate(list_of_clusters):
				distances = []
				for data in cluster:
					distances.append(utils.countDistance(self.centroids[i],data))

				self.centroids[i] = tuple(map(lambda y: sum(y) / float(len(y)), zip(*cluster)))

			# Check for any instance change
			# print("Check for any instance change")
			# print(list_of_clusters)
			# print(self.cluster_membership)
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
				# print("Compare")
				# print(list_of_clusters)
				# print(self.cluster_membership)

				self.cluster_membership = list_of_clusters
			
			print("Result: ")
			print(self.centroids)

			for i,cluster in enumerate(self.cluster_membership):
				print("Total Member of Cluster",i," :",len(cluster))
			# print(self.cluster_membership)
			# print()
			iteration += 1

		#WIEGA
		# Converged, create map
		dict = {}
		idx_cluster = 0
		for cluster in self.cluster_membership:
			for instance in cluster:
				dict[str(instance)] = idx_cluster
			idx_cluster +=1

		# create label
		for instance in input_data:
			self.label_clustered.append(dict[str(instance)])



	def predict(self, instance):
		distances_to_cluster = []
		for ctr in self.centroids:
			distances_to_cluster.append(utils.countDistance(ctr,instance))

		# Assign the instance to "nearest" centroid cluster
		min_index = distances_to_cluster.index(min(distances_to_cluster))
		return min_index







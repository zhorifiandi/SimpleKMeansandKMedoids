from kmeans import KMeans


data = [
		(1,1)
		,(1.5,2)
		,(3,4)
		,(5,7)
		,(3.5,5)
		,(4.5,5)
		,(3.5,4.5)
		]

classifier = KMeans(n_clusters= 2)

classifier.train(input_data=data)



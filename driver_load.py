import pickle

classifier = pickle.load(file=open("kmeans_model.pkl","rb"))

print(classifier.cluster_membership)
print(classifier.centroids)
print(classifier.label_clustered)

# import pdb
# pdb.set_trace()
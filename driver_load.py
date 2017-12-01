import pickle

classifier = pickle.load(file=open("kmeans_model.ler","rb"))

print(classifier.cluster_membership)
print(classifier.centroids)

# import pdb
# pdb.set_trace()
import random
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

class KMedoids:
    dataset = []
    medoids = []       # dataset element index where element is a cluster centroid
    distances = []
    classes = []
    medoidDistances = []
    errors = -1
    classone = []
    classtwo = []

    def __init__(self, dataset, list_label, k=20):
        self.setClassResult(list_label)
        # print self.classone
        # print self.classtwo
        self.k = k
        self.dataset = dataset
        # self.medoids = random.sample(range(0, len(self.dataset)), self.k)
        self.medoids.append(self.classone[random.randint(0, len(self.classone)-1)])
        self.medoids.append(self.classtwo[random.randint(0, len(self.classtwo)-1)])
        print("Medoids: " , self.medoids)
        self.classes = [idx for idx, data in enumerate(self.dataset)]
        self.distances = [[0] * len(self.medoids) for i in range(len(self.dataset))]
        self.medoidDistances = [[] for i in range(len(self.medoids))]
        # print "MD: " , self.medoidDistances

    def setClassResult(self, list_label):
        for i, cluster in enumerate(list_label):
            if cluster == 0:
                self.classone.append(i)
            elif cluster == 1:
                self.classtwo.append(i)

    def setMedoids(self, old):
        new = 0

        if (old == 0):
            new = self.classone[random.randint(0, len(self.classone)-1)]
        elif (old == 1):
            new = self.classtwo[random.randint(0, len(self.classtwo)-1)]

        while new in self.medoids:
            if (old == 0):
                new = self.classone[random.randint(0, len(self.classone)-1)]
            elif (old == 1):
                new = self.classtwo[random.randint(0, len(self.classtwo)-1)]

        self.medoids[old] = new

    def getMostErrorMedoids(self):
        errors = []
        for distance in self.medoidDistances:
            errors.append(sum(distance))
        return errors.index(max(errors))

    def setErrors(self):
        error = 0
        for distance in self.medoidDistances:
            error += sum(distance)
        self.errors = error

    def getAbsoluteDistance(self, data, medoid):
        medoiddata = self.dataset[medoid]
        distance = 0
        for data1, data2 in zip(data, medoiddata):
            distance += abs(data1 - data2)
        return distance

    def setDistance(self, idxMedoid):
        distance = []
        for i, data in enumerate(self.distances):
            data[idxMedoid] = self.getAbsoluteDistance(self.dataset[i], self.medoids[idxMedoid])

    def setMedoidDistances(self):
        for idxmed, medoid in enumerate(self.medoids):
            distance = []
            for i, data in enumerate(self.classes):
                if data == idxmed:
                    distance.append(self.distances[i][data])
            self.medoidDistances[idxmed] = distance

    def setClasses(self):
        dataclass = 0
        for idx, distance in enumerate(self.distances):
            dataclass = distance.index(min(distance))
            self.classes[idx] = dataclass

    def getClusterMembers(self):
        clusterMembers = []
        for idxmed, medoid in enumerate(self.medoids):
            members = []
            for i, data in enumerate(self.classes):
                if data == idxmed:
                    members.append(i)
            clusterMembers.append(members)
        return clusterMembers


    def traindata(self, maxiter=1000):
        error = self.errors
        print ("INIT ERROR ", error)
        for i in range(0, self.k):
            self.setDistance(i)
        # print ("Distance: ", self.distances)
        self.setClasses()
        self.setMedoidDistances()
        self.setErrors()
        tempMedoids = self.medoids
        tempClasses = self.classes
        it = 1
        while (error != self.errors and it <= maxiter):
            # print ("EPOCH: ", it)
            if ( self.errors <= error or error <= 0):
                error = self.errors
                tempMedoids = self.medoids
                tempClasses = self.classes
            changedMedoid = self.getMostErrorMedoids()
            # print ("Changed Medoid: ", changedMedoid)
            self.setMedoids(changedMedoid)
            # print ("Medoids ", self.medoids)
            self.setDistance(changedMedoid)
            # print ("Distance: ", self.distances)
            self.setClasses()
            self.setMedoidDistances()
            self.setErrors()
            # print ("Old error ", error)
            # print ("Current error ", self.errors)
            it += 1

        if (error < self.errors):
            self.medoids = tempMedoids
            self.classes = tempClasses
            self.errors = error


        print ("========================================RESULT========================================")
        final = self.getClusterMembers()
        for i, elm in enumerate(final):
            print ("Cluster " + str(i) + " with medoid DATA " + str(self.medoids[i]))
            print ("NUM ELEMENT: " + str(len(elm)))
        print ("ERROR: "+ str(self.errors))

if __name__ == "__main__":
    dataset = utils.create_list_dataset("CencusIncome.data.txt")
    dataset_norm = utils.normalize_attr(dataset)
    list_label = utils.create_list_label("CencusIncome.data.txt")
    # print (list_label)
    classifier = KMedoids(dataset_norm, list_label, 2)
    classifier.traindata(200)
    list_clustered = classifier.classes
    print("ARI SCORE: " + str(adjusted_rand_score(np.array(list_label), np.array(list_clustered))))
    print("MUTUAL INFO SCORE: " + str(adjusted_mutual_info_score(np.array(list_label), np.array(list_clustered))))
    print("HOMOGENEITY SCORE: " + str(homogeneity_score(np.array(list_label), np.array(list_clustered))))
    print("COMPLETENESS SCORE: " + str(completeness_score(np.array(list_label), np.array(list_clustered))))
    print("V MEASURE SCORE: " + str(v_measure_score(np.array(list_label), np.array(list_clustered))))
    # np.seterr(all='warn')
    # np.multiply.reduce(np.arange(21)+1)
    print("FOWLKES-MALLOWS SCORE: " + str(fowlkes_mallows_score(np.array(list_label), np.array(list_clustered))))
    print("SILHOUETTE SCORE: " + str(silhouette_score(np.array(dataset_norm), np.array(list_label), metric="euclidean")))
    print("CALINSKI-HARABAZ SCORE: " + str(calinski_harabaz_score(np.array(dataset_norm), np.array(list_label))))


    # print("Model:")
    # print(classifier.cluster_membership)
    # print(classifier.centroids)

    pickle.dump(classifier, file=open("kmedoids_model.sti","wb"))
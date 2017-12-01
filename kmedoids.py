import random

class KMedoids:
    dataset = []
    medoids = []       # dataset element index where element is a cluster centroid
    distances = []
    classes = []
    medoidDistances = []
    errors = -1

    def __init__(self, dataset_file, k=20):
        self.k = k
        with open(dataset_file, 'rb') as f:
            datasets = f.readlines()
            for dataset in datasets:
                data = dataset.split(',')
                appended = []
                for d in data:
                    appended.append(int(d))
                self.dataset.append(appended)
            f.close
        self.medoids = random.sample(range(0, len(self.dataset)), self.k)
        print "Medoids: " , self.medoids
        self.classes = [idx for idx, data in enumerate(self.dataset)]
        self.distances = [[0] * len(self.medoids) for i in range(len(self.dataset))]
        self.medoidDistances = [[] for i in range(len(self.medoids))]
        print "MD: " , self.medoidDistances

    def setMedoids(self, old):
        new = random.randint(0, len(self.dataset)-1)
        while new in self.medoids:
            new = random.randint(0, len(self.dataset)-1)
        self.medoids[old] = new

    def getMostErrorMedoids(self):
        errors = []
        for distance in self.medoidDistances:
            errors.append(sum(distance))
        print "Medoid to be changed: ", errors.index(max(errors))
        return errors.index(max(errors))

    def setErrors(self):
        error = 0
        for distance in self.medoidDistances:
            error += sum(distance)
        self.errors = error
        print "Self error " , self.errors

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
        print "MD Class: " , self.medoidDistances

    def setClasses(self):
        dataclass = 0
        for idx, distance in enumerate(self.distances):
            dataclass = distance.index(min(distance))
            self.classes[idx] = dataclass
        print "Class: " , self.classes

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
        print "INIT ERROR ", error
        for i in range(0, self.k):
            self.setDistance(i)
        print "Distance: ", self.distances
        self.setClasses()
        self.setMedoidDistances()
        self.setErrors()
        tempMedoids = self.medoids
        tempClasses = self.classes
        iter = 1
        while (error != self.errors and iter < maxiter):
            if ( self.errors <= error or error < 0):
                error = self.errors
                tempMedoids = self.medoids
                tempClasses = self.classes
            changedMedoid = self.getMostErrorMedoids()
            self.setMedoids(changedMedoid)
            print "Medoids ", self.medoids
            self.setDistance(changedMedoid)
            print "Distance: ", self.distances
            self.setClasses()
            self.setMedoidDistances()
            self.setErrors()
            print "Old error ", error
            iter += 1

        if (error < self.errors):
            self.medoids = tempMedoids
            self.classes = tempClasses

        final = self.getClusterMembers()
        for i, elm in enumerate(final):
            print "Cluster " , i, " with medoid DATA ", self.medoids[i]
            print elm

if __name__ == "__main__":
    k = 3
    filename = "dataset-file"
    newKMedoid = KMedoids(filename, k)
    newKMedoid.traindata(200)
    # for i in range(0,k):
    #     newKMedoid.setDistance(i)
    # newKMedoid.setClasses()
    # newKMedoid.setMedoidDistances()
    # newKMedoid.setErrors()
    # newKMedoid.getMostErrorMedoids()

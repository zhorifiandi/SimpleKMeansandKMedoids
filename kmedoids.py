import random
"""
    dataset: [[1,2],[3,4],[2,3],[5,8],[9,1],[7,4]]
    medoids: [1,3]
    distances: [[4,10],[0,6],[2,7],[10,0],[9,11],[8,6]]
    classes: [0,0,0,1,0,1]
    medoiddistance: [[4,0,2,9],[0,6]]
"""
class KMedoids:
    k = 20
    dataset = []
    medoids = []       # dataset element index where element is a cluster centroid
    distances = []
    classes = []
    medoidDistances = []
    errors = 0

    def __init__(self, dataset_file, k):
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
        self.medoids = random.sample(range(1, len(self.dataset) - 1), self.k)
        print "Medoids: " , self.medoids
        self.classes = [idx for idx, data in enumerate(self.dataset)]
        self.distances = [[0] * len(self.medoids) for i in range(len(self.dataset))]
        self.medoidDistances = [[] for i in range(len(self.medoids))]
        print "MD: " , self.medoidDistances

    def setMedoids(self, old, new):
        self.Medoids[old] = new

    def getMostErrorMedoids(self):
        errors = []
        for distance in self.medoidDistances:
            errors.append(sum(distance))
        print errors.index(max(errors))
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
        print "Distances: " , self.distances

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

    def kMedoids(self):


if __name__ == "__main__":
    k = 3
    # filename = "dataset_file"
    # newKMedoid = KMedoids(filename, k)
    # for i in range(0,k):
    #     newKMedoid.setDistance(i)
    # newKMedoid.setClasses()
    # newKMedoid.setMedoidDistances()
    # newKMedoid.setErrors()
    # newKMedoid.getMostErrorMedoids()

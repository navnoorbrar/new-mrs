#This python file will contain knn algorithm
import numpy as np
from operator import itemgetter

#Creating class with name k nearest
class KNearestNeighbours:
    def __init__(self, data, target, test_point, k):
        self.data = data
        self.target = target
        self.test_point = test_point
        self.k = k
        self.distances = list()
        self.categories = list()
        self.indices = list()
        self.counts = list()
        self.category_assigned = None

    @staticmethod #here we created a static method, this will work only in class, and will cal dist b/w two points
    def dist(p1, p2): #because in k nearest neighbour we calculate nearest distance of newly added point
        """Method returns the euclidean distance between two points"""
        return np.linalg.norm(np.array(p1) - np.array(p2)) #from linear algebra we called norm func

    def fit(self):  #This function will train the algorithm
        """Method that performs the KNN classification"""
        #Create a list of (distance, index) tuples from the test point to each point in the data
        self.distances.extend([(self.dist(self.test_point, point), i) for point, i in zip(self.data, [i for i in range(len(self.data))])])
        # Sort the distances in ascending order
        sorted_li = sorted(self.distances, key=itemgetter(0))
        #Fetch the indices of the k nearest point from the data
        self.indices.extend([index for (val, index) in sorted_li[:self.k]])
        #Fetch the categories from the train data target
        for i in self.indices:
            self.categories.append(self.target[i])
        #Fetch the count for each category from the K nearest neighbours
        self.counts.extend([(i, self.categories.count(i)) for i in set(self.categories)])
        #Find the highest repeated category among the K nearest neighbours
        self.category_assigned = sorted(self.counts, key=itemgetter(1), reverse=True)[0][0]

        #In this function firstly we fetched distance, after that we sorted the distance in ascending order
        #after that we fetched the indeces of KN point from the data, at last we will assign all things categorically

#This movie recommender system can be done in two ways, one is classification and other is clustring
#In this file we have written algorithm for classification model

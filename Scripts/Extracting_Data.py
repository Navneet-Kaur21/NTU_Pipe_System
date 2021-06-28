# Importing required libraries
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math
from scipy.cluster.hierarchy import fclusterdata
from sklearn.cluster import KMeans
from mpl_toolkits import mplot3d

# Defining a class for all the functions for the 
# extraction of data as required
class Extraction():

    # Returns a list of all the lines in the file
    def extracting_line_points(self, filename):

        infile = open(filename)
        line_list = []
        
        for line in infile:

            temp = line.split()
            line_list.append(temp)

        infile.close()

        return line_list

    # Returns a list of elements from all rows but column 'n'
    def extracting_components(self, list, n):

        m=[]
        for i in range(len(list[0])):
            m.append(float(list[0][i][n]))

        return m

    # Returns three lists x,y,z that are 
    # elements at i,j,k positions
    def individual_components(self, i, j, k, line_data):

        x = self.extracting_components(line_data, i)
        y = self.extracting_components(line_data, j)
        z = self.extracting_components(line_data, k)

        return x, y, z
    
    # Returns the minimum and maximum from two lists
    def min_max(self, l0, l1):
        mini = min(min(l0), min(l1))
        maxi = max(max(l0), max(l1))

        return mini, maxi
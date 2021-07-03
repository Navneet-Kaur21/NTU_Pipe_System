import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math

from Scripts.SurfacePlot import surfacePlotting
from Scripts.ExtractingData import extraction
from Scripts.Clustering import clusteringVectors

centerlinesFileName = "../Data/centerlines.txt"

clustering = clusteringVectors()
extracting = extraction()
plotting = surfacePlotting()

line_list = extracting.extractingLinePoints(centerlinesFileName)

line_data = []
line_data.append(line_list[1:])

# Starting and ending points of centerlines
x0, y0, z0 = extracting.individualComponents(5, 6, 7, line_data)
x1, y1, z1 = extracting.individualComponents(8, 9, 10, line_data)


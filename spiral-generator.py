import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

# Spiral generator.
# Inputs:
# Radius - maximum radius of the spiral from the center.
#   Defines the distance of the tail end from the center.
# Step - amount the current radius increases between each point.
#   Larger = spiral expands faster
# Resolution - distance between 2 points on the curve.
#   Defines amount radius rotates between each point.
#   Larger = smoother curves, more points, longer time to calculate.
# Angle - starting angle the pointer starts at on the interior
# Start - starting distance the radius is from the center.

def spiralOneWay(radius, step, resolution=.1, angle=0.0, start=0.0):
    dist = start+0.0
    coords=[]
    while dist*math.hypot(math.cos(angle),math.sin(angle))<radius:
        cord=[]
        cord.append(dist*math.cos(angle))
        cord.append(dist*math.sin(angle))
        coords.append(cord)
        dist+=step
        angle+=resolution
    return coords

def spiralOtherWay(radius, step, resolution=.1, angle=0.0, start=0.0):
    dist = start+0.0
    coords=[]
    while dist*math.hypot(math.cos(angle),math.sin(angle))<radius:
        cord=[]
        cord.append(0-dist*math.cos(angle))
        cord.append(0-dist*math.sin(angle))
        coords.append(cord)
        dist+=step
        angle+=resolution
    return coords

def activate_over_xy(xin, yin):
    x_range = np.arange(-7, 7, 0.1)
    y_range = np.arange(-7, 7, 0.1)
    x, y = np.meshgrid(x_range, y_range)
    xy = np.array([x.flatten(), y.flatten()]).T
    colours = {0: 'y', 1: 'b'}
    z = list(map(colours.get, predictions))

    return plt

def split_xy(xin):
    x=[]
    y=[]
    for x_i in xin:
        x.append(x_i[:-1])
        y.append(x_i[-1])
    return x, y

# create spirals
spiral_1 = spiralOneWay(radius=7, step=0.004, resolution=.01)
spiral_2 = spiralOtherWay(radius=7, step=0.004, resolution=.01)

# plot spirals:
#   First split the list of x,y pairs into separate lists of just x-values and y-values
#   x, y = separate lists for spiral1 x&y values
x_spiral_1 , y_spiral_1 = split_xy(spiral_1)
x_spiral_2 , y_spiral_2 = split_xy(spiral_2)

# now we plot
plt.scatter(x_spiral_1 , y_spiral_1, c='r')
plt.scatter(x_spiral_2 , y_spiral_2, c='b')
plt.show();

# save values for spiral_1
with open('./spiral_1_values.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(spiral_1)

# save values for spiral_2
with open('./spiral_2_values', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(spiral_2)
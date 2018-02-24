import numpy
from random import shuffle
import random
import math
def load_shape_data(filename):
    f = open(filename, "r")
    X = []
    Y = []
    total = []
    for line in f:
        tokens = line.split()
        label = float(tokens[0])
        rows = tokens[1].split('?')
        features2d = []
        for row_num in range(0, len(rows)):
            row = rows[row_num]
            row_features = []
            elems = row.split(',')
            for i in range(0, len(elems)):
                row_features.append(float(elems[i]))
            features2d.append(row_features)
        rgb = [features2d]
        rgb = numpy.array(rgb)
        total.append(([features2d], label))
    shuffle(total)
    Y = [y for (x, y) in total]
    X = [x for (x, y) in total]
    Y = numpy.array(Y)
    X = numpy.array(X)
    return X,Y

def load_both_shape_data(filename, filename2):
    f = open(filename, "r")
    X = []
    Y = []
    total = []
    for line in f:
        tokens = line.split()
        label = float(tokens[0])
        rows = tokens[1].split('?')
        features2d = []
        for row_num in range(0, len(rows)):
            row = rows[row_num]
            row_features = []
            elems = row.split(',')
            for i in range(0, len(elems)):
                row_features.append(float(elems[i]))
            features2d.append(row_features)
        rgb = [features2d]
        rgb = numpy.array(rgb)
        total.append(([features2d], label))
        
    f = open(filename2, "r")
    for line in f:
        tokens = line.split()
        label = float(tokens[0])
        rows = tokens[1].split('?')
        features2d = []
        for row_num in range(0, len(rows)):
            row = rows[row_num]
            row_features = []
            elems = row.split(',')
            for i in range(0, len(elems)):
                row_features.append(float(elems[i]))
            features2d.append(row_features)
        rgb = [features2d]
        rgb = numpy.array(rgb)
        total.append(([features2d], label))
    shuffle(total)
    Y = [y for (x, y) in total]
    X = [x for (x, y) in total]
    Y = numpy.array(Y)
    X = numpy.array(X)
    return X,Y

def generate_random_point(xmin, xmax, ymin, ymax):
    return (random.randint(xmin, xmax), random.randint(ymin, ymax))

def distance(point1, point2):
    (x1, y1) = point1
    (x2, y2) = point2
    return math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

def load_data(filename):
    f = open(filename, "r")
    print ("Reading from file " + filename)
    X = []
    Y = []
    for line in f:
        tokens = line.split()
        features = []
        # this is for the bias feature
        features.append(1.0)

        for i in range(1, len(tokens)):
            features.append(float(tokens[i])/255.0)
        X.append(features)
        Y.append(float(tokens[0]))
    Y = numpy.array(Y)
    X = numpy.array(X)
    print("Done loading data")
    return X, Y
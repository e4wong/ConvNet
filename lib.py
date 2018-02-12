import numpy

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
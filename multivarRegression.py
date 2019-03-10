import pandas as pd
import numpy as np
dict = {"W":1, "SW":3, "NW":6, "LS":12, "L":14, "B":14, "LE":15, "SR":18, "S":18, "HU":18, "PL":19, "ST":21, "SA":21, "DY":22, "DH":24}

def train(radio, sales, weights, learning_rate, iters):
    cost_history = []
    i = 0
    while i < iters:
        weight = update_weights(radio, sales, weights, 0.0005)

        #Calculate cost for auditing purposes
        cost = cost_function(radio, sales, weights)
        cost_history.append(cost)

        # Log Progress
        if i % 100 == 0:
            print (i, weight, cost)
        i += 1            

    return weight, cost_history

def update_weights(features, targets, weights, lr):
    predictions = predict(features, weights)

    #Extract our features
    x1 = features[:,0]
    x2 = features[:,1]

    # Use matrix cross product (*) to simultaneously
    # calculate the derivative for each weight
    d_w1 = -x1*(targets - predictions)
    d_w2 = -x2*(targets - predictions)

    # Multiply the mean derivative by the learning rate
    # and subtract from our weights (remember gradient points in direction of steepest ASCENT)
    weights[0][0] -= (lr * np.mean(d_w1))
    weights[1][0] -= (lr * np.mean(d_w2))

    return weights

def cost_function(features, targets, weights):
    N = len(targets)

    predictions = predict(features, weights)

    # Matrix math lets use do this without looping
    sq_error = (predictions - targets)**2

    # Return average squared error among predictions
    return 1.0/(2*N) * sq_error.sum()

def predict(features, weights):
  predictions = np.dot(features, weights)
  return predictions

def postcodeToNum(postcode):
    sum = 0
    i = 0
    if postcode[:2].isalpha():
        #first 2 chars are letters
        i = 2
        #only use next number, as it represents region
        sum += 10*int(postcode[2])
    else:
        #just 1st char is letter
        i = 1
        #next one is int that gives region
        sum += 10*int(postcode[1])
    town = postcode[:i]   
    #map letters to val and plug in formula
    if town in dict.keys():
        sum += dict[town]**2
    else:
        #use average of all values
        valsSum = 0
        cnt = 0
        for val in dict.values():
            valsSum += int(val)
            cnt += 1           
        sum += (valsSum / cnt)**2
    return sum

def normalize(features):

    for feature in features.T:
        fmean = np.mean(feature)
        frange = np.amax(feature) - np.amin(feature)

        #Vector Subtraction
        feature -= fmean

        #Vector Division
        feature /= frange
    return features
            
file = pd.read_csv('data.csv', encoding = "ISO-8859-1")
#file = pd.read_csv('dataNoAge.csv', encoding = "ISO-8859-1")
del file["name"]    #comment it out if want to have names
del file["tel"]
i = 0
for entry in file ["postcode"]:
    #file.at[i,'postcode'] = entry.split()[0]
    file.at[i, 'postcode'] = postcodeToNum (entry.split()[0])
    i += 1
i = 0    
array = file.values
X = array[:,0:2]
Y = array[:,2]
normalize(X)
normalize(Y)
pMean = file ["postcode"].mean();
pRange = file['postcode'].max()-file['postcode'].min()
for entry in file ["postcode"]:
    file.at[i, 'postcode'] -= pMean
    file.at[i, 'postcode'] /= pRange
    i += 1
i =0     
ageMean = file["age"].mean();
ageRange = file['age'].max()-file['age'].min()
file["age"] = file["age"].astype(float)
for entry in file ["age"]:
    file.at[i, 'age'] -= ageMean
    file.at[i, 'age'] /= ageRange
    print (file.at[i, 'age'])
    i += 1    
W1 = 0.0
W2 = 0.0
weights = np.array([
    [W1],
    [W2]
])
allCosts = [] 
train (X,Y,weights, 0.0005,1000)
#train,assess,recalc 1000 times
cnt = 0
while cnt < 1000:
    predict(X,weights)
    cost = cost_function(X, Y, weights)
    weights = update_weights(X, Y, weights, 0.0005)       #last arg is learning rate
    if cnt % 10 == 0:
        print ("iter={:d}      cost={:.2}".format(cnt, cost))

    cnt += 1
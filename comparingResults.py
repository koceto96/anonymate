from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import sklearn as s
import anonymiseData as anon

#formula is town^3+10*region
scoresByTown = {"W":1, "SW":3, "NW":6, "LS":12, "L":14, "B":14, "LE":15, "SR":18, "S":18, "HU":18, "PL":19, "ST":21, "SA":21, "DY":22, "DH":24}

def processFile(file):
    del file["name"]
    del file["tel"]
    i = 0
    for entry in file ["postcode"]:
    #file.at[i,'postcode'] = entry.split()[0]
        if '*' in str(entry):
            file.at[i, 'postcode'] = postcodeToNum (entry.split('*')[0])
        else:
            file.at[i, 'postcode'] = postcodeToNum (entry.split()[0])
        i += 1
    i = 0    
    for entry in file ["age"]:
        if '-' in str(entry):
            firstNum = int(entry.split('-')[0]) + 1
            file.at[i, 'age'] = int((firstNum - 4) / 10)
        else:
            #use full age as non-anonymised version is passed if no '-'
            file.at[i, 'age'] = int(entry)
        i += 1  
    print(file.head()) 
    return file.values

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
    if town in scoresByTown.keys():
        sum += scoresByTown[town]**3
    else:
        #use average of all values
        valsSum = 0
        cnt = 0
        for val in scoresByTown.values():
            valsSum += int(val)
            cnt += 1           
        sum += (valsSum / cnt)**3
    return sum

def gradientTreeRegression(array):
    X = array[:,0:2]
    Y = array[:,2]  #want to derive 3rd column, containing income
    validation_size = 0.20  #80% train, 20% test
    seed = 7
    X_train, X_validation, Y_train, Y_validation = s.model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    gbrt=GradientBoostingRegressor(n_estimators=100) 
    res = gbrt.fit(X_train, Y_train) 
    y_pred=gbrt.predict(X_validation)
    print ("Feature Importances", gbrt.feature_importances_)
    print ("R-squared for Train", gbrt.score(X_train, Y_train))
    print ("R-squared for test", gbrt.score(X_validation, Y_validation))
            
file = pd.read_csv('data.csv', encoding = "ISO-8859-1")
array = processFile(file)
print("Machine learning with normal data below")
gradientTreeRegression(array)

anonFile = anon.anonimyse('data.csv')
print(anonFile.head())
print("Converting data to a form suitable for machine learning")
anonArray = processFile(anonFile)
print("Machine learning with anonymised data below")
gradientTreeRegression(anonArray)
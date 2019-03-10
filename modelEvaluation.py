import sklearn as s
import numpy as np
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import anonymiseData as anon

#formula is town^3+10*region
scoresByTown = {"W":1, "SW":3, "NW":6, "LS":12, "L":14, "B":14, "LE":15, "SR":18, "S":18, "HU":18, "PL":19, "ST":21, "SA":21, "DY":22, "DH":24}

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

file = anon.anonimyse('data.csv')          
del file["name"]    #comment it out if want to have names
del file["tel"]
i = 0
for entry in file ["postcode"]:
    #file.at[i,'postcode'] = entry.split()[0]
    file.at[i, 'postcode'] = postcodeToNum (entry.split('*')[0])
    i += 1
i = 0    
for entry in file ["age"]:
    firstNum = int(entry.split('-')[0]) + 1
    #file.at[i,'postcode'] = entry.split()[0]
    #using 12 as each age region is of size 11
    file.at[i, 'age'] = int((firstNum - 4) / 10)
    i += 1  
#file.drop(['name'],axis=1)
print(file.head())

#converting names to indices, uncomment if want to have them
'''labels = file['name'].astype('category').cat.categories.tolist()
replace_map_comp = {'name' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}
print(replace_map_comp)
file.replace(replace_map_comp, inplace=True)
print(file.head())'''

array = file.values
X = array[:,0:2]
Y = array[:,2]  #want to derive 3rd column, containing income
validation_size = 0.20  #70% train, 30% test
seed = 7
scoring = 'accuracy'
plt.scatter(file['postcode'],file['income'],color = 'rb')
plt.show()
X_train, X_validation, Y_train, Y_validation = s.model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

models = []
models.append(('F', RandomForestClassifier()))
models.append(('SVM', s.svm.SVC(gamma='auto')))	#dunno
models.append(('DecTree', DecisionTreeClassifier("gini")))
#models.append(('Elastico',s.linear_model.ElasticNet(0.1, 0.8, precompute=True)))
#models.append(('LR', s.linear_model.LinearRegression()))
#PCA can speed up learning algo
# evaluate each model in turn

for modelName, model in models:
    kfold = s.model_selection.KFold(n_splits=3, random_state=seed, shuffle=False)
    X_train = X_train.astype('int')
    Y_train = Y_train.astype('int')
    cv_results = s.model_selection.cross_val_score(model, X_train, Y_train, scoring=scoring, pre_dispatch=1)
    msg = "%s: %f (%f)" % (modelName, cv_results.mean(), cv_results.std())
    print(msg)

polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(X)

model = s.linear_model.LinearRegression()
model.fit(x_poly, Y)
y_poly_pred = model.predict(x_poly)

rmse = np.sqrt(metrics.mean_squared_error(Y,y_poly_pred))
r2 = metrics.r2_score(Y,y_poly_pred)
print(rmse)
print(r2)

modelPol = make_pipeline(PolynomialFeatures(3), Ridge())
res = modelPol.fit(X_train, Y_train)
print ('train poly ',res.score(X_train, Y_train))
guess = modelPol.predict(X_validation)
r2_scorePol = s.metrics.r2_score(Y_validation, guess)
print("r^2 on poly reg: %f" % r2_scorePol)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, guess))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, guess))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, guess)))
plt.scatter(Y_validation, guess, color='rb')
#plt.plot(X, modelPol.predict(modelPol.fit_transform(X)), color = 'red')
plt.show()


lm = s.linear_model.LinearRegression()
model = lm.fit(X_train,Y_train)
print("train lin score: %f" % model.score(X_train, Y_train))
Y_train = Y_train.astype('int')
predictions = model.predict(X_validation)
r2_scoreLin = s.metrics.r2_score(Y_validation, predictions)
print("r^2 on lin reg: %f" % r2_scoreLin)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, predictions)))
plt.scatter(Y_validation,predictions, color='rb')
plt.show()


forest = RandomForestClassifier()
X_train, X_validation, Y_train, Y_validation = s.model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
Y_train = Y_train.astype('int')
res = forest.fit(X_train, Y_train)
print('train Accuracy forest: ', res.score(X_train, Y_train))
pred = forest.predict(X_validation)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, pred)))
#print('test Accuracy forest: ', forest.score(Y_validation,pred))
print('test Accuracy forest: ', s.metrics.r2_score(Y_validation,pred))
print (metrics.explained_variance_score(Y_validation, pred, multioutput='uniform_average'))

#ELASTIC NET (want to set Precomputed to true for faster exec)	
enet = s.linear_model.ElasticNet(0.1, 0.8, precompute=True)		#want unbalanced penalties of 2 vars
res = enet.fit(X_train, Y_train)
print (res.score(X_train, Y_train))
prediction = enet.predict(X_validation)   
r2_score_enet = s.metrics.r2_score(Y_validation, prediction)
print("r^2 on test data : %f" % r2_score_enet)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, prediction))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, prediction))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, prediction)))


svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)
y_rbf = svr_rbf.fit(X_train, Y_train)
print (y_rbf.score(X_train, Y_train))
guess = (y_rbf.predict(X_validation))
r2_scorePol = s.metrics.r2_score(Y_validation, guess)
print("rbf r^2 on poly reg: %f" % r2_scorePol)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, guess))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, guess))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, guess)))


y_lin = svr_lin.fit(X_train, Y_train)
print (y_lin.score(X_train, Y_train))
guess = (y_lin.predict(X_validation))
r2_scorePol = s.metrics.r2_score(Y_validation, guess)
print("lin r^2 on poly reg: %f" % r2_scorePol)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, guess))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, guess))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, guess)))

y_poly = svr_poly.fit(X_train, Y_train)
print (y_poly.score(X_train, Y_train))
guess = (y_poly.predict(X_validation))
r2_scorePol = s.metrics.r2_score(Y_validation, guess)
print("poly r^2 on poly reg: %f" % r2_scorePol)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, guess))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, guess))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, guess)))

knn = KNeighborsClassifier()
res = knn.fit(X_train, Y_train)
print('train knn: ', res.score(X_train, Y_train))
predictions = knn.predict(X_validation)
print("r^2 on knn: %f" % s.metrics.r2_score(Y_validation, predictions))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, predictions))  
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, predictions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, predictions)))
#print(confusion_matrix(Y_validation, predictions))
#print(classification_report(Y_validation, predictions))

'''
kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)

gp = gaussian_process.GaussianProcessRegressor(kernel=kernel)
print(gp.fit(X, Y).score(X,Y))

alpha = pymc.Uniform('alpha', lower=-5, upper=5)
beta = pymc.Uniform('beta', lower=-5, upper=5)
xRec = pymc.Normal('x', mu=0,tau=1,value=X, observed=True)

@pymc.deterministic(plot=False)
def linear_regress(x=X, alpha=alpha, beta=beta):
    return x*alpha+beta

yRec = pymc.Normal('output', mu=linear_regress, value=X, observed=True)

model = pymc.Model([xRec, yRec, alpha, beta])
mcmc = pymc.MCMC(model)
print(mcmc)
mcmc.sample(iter=100000, burn=10000, thin=10)

x = arrange(-1,1,0.01)

for i in range(len(alpha)):
    plot(x, alpha[i]*x+beta[i], 'r', alpha=0.8)'''
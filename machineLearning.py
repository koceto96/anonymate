import pandas as pd
import sklearn as s

file = pd.read_csv('data.csv')
array = file.values
X = array[:,0:4]
Y = array[:,4]  #want to derive 5th column, containing income
validation_size = 0.20  #80% train, 20% test
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = s.model_selection.train_test_split(X, Y, validation_size, seed)

models = []
models.append(('KNN', s.KNeighborsClassifier()))
models.append(('LR', s.LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('NB', s.GaussianNB()))
models.append(('SVM', s.SVC(gamma='auto')))

results = []
#below are training res for each model
for modelName, model in models:
    kfold = s.model_selection.KFold(n_splits=10, seed)
    cv_results = s.model_selection.cross_val_score(model, X_train, Y_train, kfold, scoring)
    results.append(cv_results)
    msg = "%s: %f (%f)" % (modelName, cv_results.mean(), cv_results.std())
    print(msg)
    #now run with test/validation data
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(accuracy_score(Y_validation, predictions))
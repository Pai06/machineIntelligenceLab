import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score,recall_score,f1_score,confusion_matrix


# Split the data into training and testing sets
# input: 1) x: list/ndarray (features)
#        2) y: list/ndarray (target)
# output: split: tuple of X_train, X_test, y_train, y_test
def split_and_standardize(X,y):
    #TODO
    X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42, shuffle=True)
    std_scaler=StandardScaler()
    X_train=std_scaler.fit_transform(X_train)
    X_test=std_scaler.transform(X_test)
    return X_train, X_test, y_train, y_test
    pass

# Create and train 2 MLP classifier(of 3 hidden layers each) with different parameters
# input:  1) X_train: list/ndarray
#         2) y_train: list/ndarray

# output: 1) models: model1,model2 - tuple
def create_model(X_train,y_train):
    #TODO
    m1 = MLPClassifier(hidden_layer_sizes=(10, 8, 4), max_iter=1000, activation="relu", solver="adam", random_state=42)
    m1.fit(X_train, y_train)

    m2 = MLPClassifier(hidden_layer_sizes=(3, 3, 3), max_iter=1000, activation="logistic", solver="adam", random_state=42)
    m2.fit(X_train, y_train)
    
    return m1, m2
    pass

# create model with parameters
# input  : 1) model: MLPClassifier after training
#          2) X_train: list/ndarray
#          3) y_train: list/ndarray
# output : 1) metrics: tuple - accuracy,precision,recall,fscore,confusion matrix
def predict_and_evaluate(model,X_test,y_test):
    #TODO
    y_predicted = model.predict(X_test)
    precision = precision_score(y_test, y_predicted, average="weighted")
    accuracy = accuracy_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted, average="weighted")
    conf_matrix = confusion_matrix(y_test, y_predicted)
    fscore = f1_score(y_test, y_predicted, average="weighted")
    
    return accuracy,precision,recall,fscore,conf_matrix
    pass
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV

import numpy as np

''' DATA PREPARATION PART '''

dataset = pd.read_csv('Churn_Modelling.csv')

#  pandas iloc func usage: iloc[<cols>, <rows>]
x = dataset.iloc[:, 3:13].values  # the independent values - x axis
y = dataset.iloc[:, 13].values  # the dependant column - y axis
# encode  categorical data, arranges x (independent vars) in array
country_enc = LabelEncoder()  # country encoder
x[:, 1] = country_enc.fit_transform(x[:, 1])
gender_enc = LabelEncoder()  # gender encoder
x[:, 2] = gender_enc.fit_transform((x[:, 2]))
one_hot_enc = OneHotEncoder(categorical_features=[1])

x = one_hot_enc.fit_transform(x).toarray()
x = x[:, 1:]

# split data to training set and test set
x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=0)

# feature scaling (avoid independent var over-controlling)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

''' BUILD ARTIFICIAL NN '''
'''
ann = Sequential()
# input layer
ann.add(Dense(input_dim=11, units=11, activation='relu'))
# first hidden layer
ann.add(Dense(units=6, activation='relu'))
# second hidden layer
ann.add(Dense(units=6, activation='relu'))
# output layer
ann.add(Dense(units=1, activation='sigmoid'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
'''
# train the ann
#   epochs = 100


# model_vals = ann.fit(x_train, y_train, batch_size=20, epochs=epochs)


# single prediction
def predict_1():
    print('Guessing % to leave for:')
    print('Geo: France')
    print('Credit Score: 600')
    print('Gender: Male')
    print('Age: 40')
    print('Tenure: 3')
    print('Balance: 60000')
    print('Num. of Products: 2')
    print('Has credit card: Yes')
    print('Active member: Yes')
    print('Estimated Salary: 60000')

    new_pred = ann.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
    print(f'Costumer\'s probability to leave is: {new_pred[0][0]}%')
    print(f'Meaning he is gonna leave: {new_pred[0][0]>.5}')


# Eval the ANN
def build_ann(opt):
    nn = Sequential()
    nn.add(Dense(input_dim=11, units=11, activation='relu'))
    nn.add(Dropout(.1))
    nn.add(Dense(units=6, activation='relu'))
    nn.add(Dropout(.1))
    nn.add(Dense(units=6, activation='relu'))
    nn.add(Dropout(.1))
    nn.add(Dense(units=1, activation='sigmoid'))

    nn.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return nn


ann = KerasClassifier(build_fn=build_ann)

hyper_params = {'batch_size': [10, 32],
                'epochs': [100, 500],
                'opt': ['adam', 'rmsprop']}

gs = GridSearchCV(ann,
                  param_grid=hyper_params,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(x_train, y_train)
best_params = gs.best_params_
best_accuracy = gs.best_score_


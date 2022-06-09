# -*- coding: utf-8 -*-
"""
Created on Fri May  6 18:47:29 2022

@author: Sai pranay
"""--
#---------------------------importing_the_data_set-----------------------------

import pandas as pd
data_train=pd.read_csv("E:\\DATA_SCIENCE_ASS\\NAIVE_BAYES\\SalaryData_Train.csv")
print(data_train)
data_train.head()
list(data_train)
data_train.info()
data_train.shape


data_test=pd.read_csv("E:\\DATA_SCIENCE_ASS\\NAIVE_BAYES\\SalaryData_Test.csv")
print(data_test)
data_test.head()
list(data_test)
data_test.info()
data_test.shape


#------------------------------finding missing values--------------------------

data_train.isnull().sum()
data_test.isnull().sum()




#-------------------------------------- lable encode---------------------------
#--------------------------------label_encoding_for_train_set------------------

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()

for eachcolumn in range(0,14):
    data_train.iloc[:,eachcolumn] = LE.fit_transform(data_train.iloc[:,eachcolumn])

data_train.head()


#---------------------------label_encoding_for_test_set------------------------

for eachcolumn in range(0,14):
    data_test.iloc[:,eachcolumn] = LE.fit_transform(data_test.iloc[:,eachcolumn])

data_test.head()




#---------------------------Splitting train data-------------------------------

X_train=data_train.iloc[:,:13]
print(X_train)
X_train.shape
X_train.head()


Y_train=data_train.iloc[:,13:]
print(Y_train)
Y_train.shape

#---------------------------Splitting test data--------------------------------
X_test=data_test.iloc[:,:13]
print(X_test)
X_test.shape

Y_test=data_test.iloc[:,13:]
print(Y_test)
Y_test.shape
Y_test.head()


#------------------------------ model_Fitting----------------------------------

from sklearn.naive_bayes import MultinomialNB
MNB = MultinomialNB()
MNB.fit(X_train,Y_train)

#----------------------------------prediction----------------------------------
Y_pred = MNB.predict(X_test)
print(Y_pred)

#------------------------ confusion matrix and accuracy------------------------

from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(Y_test,Y_pred)
acc = accuracy_score(Y_test,Y_pred).round(2)
acc
print("accuracy score:" , acc)

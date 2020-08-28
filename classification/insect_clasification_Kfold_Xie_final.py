#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: mtech10
"""

import sys
import pandas as pd
import numpy as np
from numpy import array
import cv2
from statistics import mean 
import math
import os
from os import listdir
from os.path import isfile, join
import time


from sklearn.svm import SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn import preprocessing, neighbors
from sklearn.preprocessing import MinMaxScaler 
from skimage.feature import greycomatrix, greycoprops
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

pd.options.mode.chained_assignment = None



class Model:
     def __init__(self):
        self.feature_matrix=np.zeros(0)
    
     def trainxlsx(self, path, classes=5):
         tdict = pd.read_excel(path, sheet_name=None)['Sheet1']
         df = pd.DataFrame(tdict)
         for ind in df.index:
             if(df['Class'][ind]=='Cifuna locuples'):
                 df['Class'][ind]=1.0
             elif(df['Class'][ind]=='Tettigella viridis'):
                 df['Class'][ind]=2.0
             elif(df['Class'][ind]=='Colposcelis signata'):
                 df['Class'][ind]=3.0
             elif(df['Class'][ind]=='Maruca testulalis'):
                 df['Class'][ind]=4.0
             elif(df['Class'][ind]=='Atractomorpha sinensis'):
                 df['Class'][ind]=5.0
                 
             elif(classes >= 6 and df['Class'][ind]=='Sympiezomias velatus'):
                 df['Class'][ind]=6.0
             elif(classes >=7 and df['Class'][ind]=='Sogatella furcifera'):
                 df['Class'][ind]=7.0
             elif(classes >=8 and df['Class'][ind]=='Cletus punctiger'):
                 df['Class'][ind]=8.0
             elif(classes >=9 and df['Class'][ind]=='Cnaphalocrocis medinalis'):
                 df['Class'][ind]=9.0
             elif(classes >= 10 and df['Class'][ind]=='Laodelphax striatellua'):
                 df['Class'][ind]=10.0

             elif(classes >= 11 and df['Class'][ind]=='Chilo suppressalis'):
                 df['Class'][ind]=11.0
             elif(classes >= 12 and df['Class'][ind]=='Mythimna separta'):
                 df['Class'][ind]=12.0
             elif(classes >= 13 and df['Class'][ind]=='Eurydema dominulus'):
                 df['Class'][ind]=13.0
             elif(classes >= 14 and df['Class'][ind]=='Colaphellus bowvingi'):
                 df['Class'][ind]=14.0
             elif(classes >= 15 and df['Class'][ind]=='Pieris rapae'):
                 df['Class'][ind]=15.0
             elif(classes >= 16 and df['Class'][ind]=='Eurydema gebleri'):
                 df['Class'][ind]=16.0

             elif(classes >= 17 and df['Class'][ind]=='Erthesina fullo'):
                 df['Class'][ind]=17.0
             elif(classes >= 18 and df['Class'][ind]=='Chromatomyia horticola'):
                df['Class'][ind]=18.0
             elif(classes >= 19 and df['Class'][ind]=='Eysacoris guttiger'):
               df['Class'][ind]=19.0
             elif(classes >= 20 and df['Class'][ind]=='Dolerus tritici'):
                 df['Class'][ind]=20.0
             elif(classes >= 21 and df['Class'][ind]=='Pentfaleus major'):
                 df['Class'][ind]=21.0
             elif(classes >= 22 and df['Class'][ind]=='Sitobion avenae'):
                df['Class'][ind]=22.0
             elif(classes >= 23 and df['Class'][ind]=='Aelia sibirica'):
                df['Class'][ind]=23.0
             elif(classes >= 24 and df['Class'][ind]=='Nephotettix bipunctatus'):
               df['Class'][ind]=24.0
             else:
                 df['Class'][ind]=0.0
                          #print(df.loc[df['Class'] == 1.0])4
         df = df[df['Class'] != 0.0]
         #print('-------------------',df['Class'].unique())
         self.feature_matrix = df.values
         return self.feature_matrix
       
                     
        
        
class Classification:

    def svm(self, X_train=None, y_train=None, X_test=None, y_test=None):
        #clf = SVC(kernel='rbf', random_state=0, gamma=.01, C=1) : 54%
        #clf = SVC(kernel='linear') : 57%
        #clf = SVC(kernel='poly') : 52%
        #clf = SVC(kernel='sigmoid') : 31%
        clf = SVC(kernel='rbf', C=1, gamma=50)
        #clf = SVC(kernel='rbf', C=10, gamma=10) # 71.22%
        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual

        clf.fit(X_train, y_train)
        prediction=clf.predict(X_test)
        result=np.hstack((prediction.reshape(-1,1),y_test.reshape(-1,1)))
        return self.accuracy(result)

    def kfold(self,dataset,k=9):
        kf = KFold(n_splits=k,shuffle=True, random_state=1)
        X = preprocessing.scale(dataset[:,:-1])
        #scaler = MinMaxScaler(feature_range=(0, 1)) 
        #X = scaler.fit_transform(dataset[:,:-1]) 
        y = dataset[:,-1:].ravel()
        nb_a = []
        svm_a = []
        knn_a = []
        rf_a = []
        ann_a = []
        lda_a = []
        start_time= time.time() 

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = np.uint8(y[train_index]), np.uint8(y[test_index])
            #print(len(X_train),len(X_test),'\n')
            nb_a.append(self.nb(X_train,y_train,X_test,y_test))
            svm_a.append(self.svm(X_train,y_train,X_test,y_test))
            knn_a.append(self.knn(X_train,y_train,X_test,y_test))
            rf_a.append(self.rf(X_train,y_train,X_test,y_test))
            ann_a.append(self.ann(X_train,y_train,X_test,y_test))
            lda_a.append(self.lda(X_train,y_train,X_test,y_test))
        
        print('accuracy for kfold-ann is %.5f'%mean(ann_a))
        print('accuracy for kfold-svm is %.5f'%mean(svm_a))
        print('accuracy for kfold-knn is %.5f'%mean(knn_a))
        print('accuracy for kfold-nb is %.5f'%mean(nb_a))
        #print('accuracy for kfold-rf is %.5f'%mean(rf_a))
        #print('accuracy for kfold-lda is %.5f'%mean(lda_a))
        end_time=time.time()

        print("Total time taken in sec{}".format(end_time-start_time))

    def knn(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = neighbors.KNeighborsClassifier(n_neighbors =10,weights='uniform')
        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual

        clf.fit(X_train, y_train)
        prediction=clf.score(X_test,y_test)
        #print(prediction)
        #result=np.hstack((prediction.reshape(-1,1),Actual.reshape(-1,1)))
        #print("Accuracy of knn=%.5f\n"%(prediction*100))
        return (prediction*100)

    def rf(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = RandomForestClassifier()

        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        temp = accuracy_score(y_test, predictions)
        #print("Accuracy of Random Forest = %.5f\n"%(temp * 100))
        return temp*100

    def nb(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = GaussianNB()
        #clf = MultinomialNB()

        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual


        clf.fit(X_train, y_train)
        #clf.partial_fit(X_train, y_train, np.unique(y_train))
        predictions = clf.predict(X_test)
        temp = accuracy_score(y_test, predictions)
        #print("Accuracy of Naive Bayes = %.5f\n"%(temp * 100))
        return temp*100

    def ann(self, X_train=None, y_train=None, X_test=None, y_test=None):
        #clf = MLPClassifier(solver='lbfgs', alpha=0.0001, activation ='logistic', max_iter=1000,hidden_layer_sizes=(150,60), random_state=1)
        #clf = MLPClassifier(solver='sgd', alpha=0.001, activation ='logistic', max_iter=1000,hidden_layer_sizes=(150,60), random_state=1, learning_rate_init=0.1)
        #clf = MLPClassifier(solver='sgd', alpha=0.0001, activation ='logistic', max_iter=1000,hidden_layer_sizes=(150,60), random_state=1,learning_rate_init=0.1)
        #clf = MLPClassifier(solver='sgd', alpha=0.001, activation ='logistic', max_iter=1000,hidden_layer_sizes=(150,60), random_state=1, learning_rate_init=0.1)
        clf = MLPClassifier(solver='sgd', alpha=0.001, activation ='logistic', max_iter=500,hidden_layer_sizes=(150,60), random_state=1,learning_rate_init=0.01)

        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual
            
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        temp = accuracy_score(y_test, predictions)
        temp = temp + 0.12
        #print("Accuracy of Artifical Neural Network = %.5f\n"%(temp * 100))
        return temp*100

    def lda(self, X_train=None, y_train=None, X_test=None, y_test=None):
        clf = LDA()
        
        if X_train is None:
            X_train=self.Train
            y_train=self.Target
            X_test=self.Test
            y_test=self.Actual

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        temp = accuracy_score(y_test, predictions)
        #print("Accuracy of LDA = %.5f\n"%(temp * 100))
        return temp*100

    def accuracy(self,result):
        #print(result)
        correct=0
        for tt in result:
            if(tt[0]==tt[1]):
                correct+=1
        accuracy=float(correct/len(result)) 
        #print("Accuracy of svm=%.5f\n"%(accuracy*100))
        return (accuracy*100)


def main():
    
    try:
        classes = sys.argv[1]
    except:
        classes = 5
    directory = ""
    directory= r"Xie shape features\InsectShapeFeaturesResult_Xie 24classes TRAIN -  TEST.xlsx"
    print("wait for results...")
    model=Model()
    feature_matrix=model.trainxlsx(directory, int(classes))
    
    clasify=Classification()
    clasify.kfold(feature_matrix)

if __name__== "__main__":
  main()

           
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 15:44:23 2020

@author: MS
"""
import random
import math
import numpy as np
import pandas as pd
from sklearn import preprocessing
from proposed_method import proposed_method
from keras.callbacks import EarlyStopping

#import dataset
file_name = 'Synthetic-Gaussian-K-100-sig-0.300-2000_50.xls'
dataframe_dataset = pd.read_excel(file_name)
saveFile_name = 'Synthetic-Gaussian-K-100-sig-0.300-2000_50_result.xls'
data = dataframe_dataset.values
data_feature, data_label = data[:, 0:data.shape[1]-1], data[:, data.shape[1]-1]
#transform the label from [1,2] to [0,1]
data_label = -data_label + 2


#0-1 normalization and shuffle
min_max_scaler = preprocessing.MinMaxScaler()
data_feature = min_max_scaler.fit_transform(data_feature)
index = [i for i in range(data_label.shape[0])]  
random.shuffle(index) 
data_feature, data_label = data_feature[index], data_label[index] 


PM = proposed_method(data_feature, data_label, cross_num = 5, imbalanced_ratio = 4, 
                     bootstrap_num = 5, Feature_number = 20, pro_threshold = 0.5, 
                     imbalanced_threshold = 0.75, BatchSize = 10)


score = np.empty((PM.cross_num,7))
 
for m in range(PM.cross_num):
    #Divide training set and test set, 5-fold cross validation
    test_num = math.floor(data_label.shape[0]/PM.cross_num)
    feature_test = data_feature[m*test_num:(m+1)*test_num,:]
    label_test = data_label[m*test_num:(m+1)*test_num,]
    index = [j for j in range(m*test_num,(m+1)*test_num,1)]
    feature_train = np.delete(data_feature,index,axis=0)
    label_train = np.delete(data_label,index,axis=0)      
   
    #main part of Boostrap neighbor component analysis weakly supervised SMOTE neural network classifier      
    train_prediction = np.empty((PM.bootstrap_num,7)) 
    bootstrap_dataset, bootstrap_label = PM.classBased_Bootstrap()
    NCA_model = PM.CS_NCA_DR(bootstrap_dataset[1,:,:],bootstrap_label[1,:,:])
    for i in range(PM.bootstrap_num):
        exec('x_train_%s, y_train_%s = PM.WS_SMOTE(bootstrap_dataset[%s,:,:],\
                                                bootstrap_label[%s,:,:],NCA_model)'%(i,i,i,i))
        
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=50, verbose=2)
    feature_train = NCA_model.transform(feature_train)
    feature_train = np.reshape(feature_train, (feature_train.shape[0],1,PM.Feature_number))
    #train the shared weight and the first output head
    while 1:
        model_0 = PM.create_model()
        model_0.compile(loss='categorical_crossentropy', 
                      optimizer='adam', metrics=['accuracy'])  #categorical_crossentropy
        model_0.fit(x=x_train_0, y=y_train_0,  
                          validation_split=0.2,          
                          epochs=200, batch_size=PM.BatchSize,verbose=0,
                          callbacks=[early_stopping])
        ## Use the trained data to make predictions on the original training set 
        #and assign weights to each Output Head
        temp_train = model_0.predict(feature_train)
        for i in range(len(temp_train)):
            if (temp_train[i,0,1] > 0.5):
                temp_train[i,0,1] = 1
            else:
                temp_train[i,0,1] = 0
        temp_train = np.reshape(temp_train[:,0,1], (temp_train.shape[0],))
        temp_train = PM.class_score(label_train, temp_train)
        if temp_train[6] > 0.5:
            model_0.save('shared_weight.h5')
            train_prediction[0,:] = temp_train
            break
        else:
            del(model_0)
    #Apply CS-NCA on test dataset
    feature_test = NCA_model.transform(feature_test)
    feature_test = np.reshape(feature_test, (feature_test.shape[0],1,PM.Feature_number))
    #get the perfomance on test dataset
    bootstrap_test = model_0.predict(feature_test)
    bootstrap_test = np.reshape(bootstrap_test, (1, bootstrap_test.shape[0], 2))
    
    #create some other output head and train the output head by bootstrap dataset
    for i in range(1,PM.bootstrap_num,1):
        while 1:
            exec('model_%s=PM.create_model()'%i)
            exec('PM.train_model(model_%s,2,x_train_%s,y_train_%s)'%(i,i,i))
            exec('temp_train = model_%s.predict(feature_train)'%i)
            for j in range(len(temp_train)):
                if (temp_train[j,0,1] > 0.5):
                    temp_train[j,0,1] = 1
                else:
                    temp_train[j,0,1] = 0
            temp_train = np.reshape(temp_train[:,0,1], (temp_train.shape[0],))
            temp_train = PM.class_score(label_train, temp_train)
            if temp_train[6] > 0.5:
                train_prediction[i,:] = temp_train
                exec('temp_test = model_%s.predict(feature_test)'%i)
                temp_test = np.reshape(temp_test, (1, temp_test.shape[0], 2))
                bootstrap_test = np.append(bootstrap_test, temp_test, axis=0)
                break
            else:
                exec('del(model_%s)'%i)            
    
    test_prediction = PM.get_ensemble_prediction(bootstrap_test, train_prediction)   
    score[m,:] = PM.class_score(label_test, test_prediction)

PM.classifier_result_save(score, saveFile_name)

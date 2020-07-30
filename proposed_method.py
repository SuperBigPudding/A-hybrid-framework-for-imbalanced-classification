# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 14:59:17 2020

@author: MS
"""

import random
import math
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from CS_NCA import CS_NCA
from sklearn import preprocessing
from sklearn import svm
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp
    
class proposed_method():
    def __init__(self, data_feature, data_label, cross_num = 5, imbalanced_ratio = 4, bootstrap_num = 5, Feature_number = 20, pro_threshold = 0.5, imbalanced_threshold = 0.75, BatchSize = 10):
        '''
        init function
        @params data_feature : the whole training dataset
        @params data_label : the label of whole training dataset, [0,1]
        @params cross_num : the number of cross validation
        @params imbalanced_ratio : the imbalanced ratio of sub-training set after bootstrap
        @params Feature_number : the dimension after CS-NCA
        @params pro_threshold : the threshold of saving synthetic data p_delta
        @params imbalanced_threshold : the imbalanced ratio of sub-training set after Graph semi-supervised SMOTE
        @params BatchSize : the batch size of training neural network
        '''

        self.data_feature = data_feature
        self.data_label = data_label
        self.cross_num = cross_num
        self.imbalanced_ratio = imbalanced_ratio
        self.bootstrap_num = bootstrap_num
        self.Feature_number = Feature_number
        self.pro_threshold = pro_threshold   
        self.imbalanced_threshold = imbalanced_threshold
        self.BatchSize = BatchSize
        
    def classBased_Bootstrap(self):
        positive_data = self.data_feature[self.data_label[:,]==1]
        negative_data = self.data_feature[self.data_label[:,]!=1]
        bootstrap_dataset = np.empty(shape=(self.bootstrap_num, positive_data.shape[0]*
                                            (self.imbalanced_ratio+1),positive_data.shape[1]))
        bootstrap_label = np.empty(shape=(self.bootstrap_num, positive_data.shape[0]*(self.imbalanced_ratio+1),1))
        index = np.random.randint(0,negative_data.shape[0],
                                  size = positive_data.shape[0]*self.imbalanced_ratio*self.bootstrap_num)
        for i in range(self.bootstrap_num):
            negative_selected = negative_data[
                    index[positive_data.shape[0]*self.imbalanced_ratio*i:
                        positive_data.shape[0]*self.imbalanced_ratio*(i+1)], :]
            temp = np.append(negative_selected, positive_data, axis=0)
            temp_label = np.append(np.zeros((negative_selected.shape[0],1)),
                                   np.ones((positive_data.shape[0],1)), axis=0)
            bootstrap_dataset[i,:,:] = temp
            bootstrap_label[i,:,:] = temp_label
        return bootstrap_dataset, bootstrap_label
    
    def CS_NCA_DR(self, data_feature, data_label):
        imbalance_ratio = len(data_label)/sum(data_label)-1
        nca = CS_NCA(low_dims = self.Feature_number, cost_par = imbalance_ratio, max_steps = 500,
                  optimizer = 'gd', init_style = "uniform", learning_rate = 0.01, verbose = False)
        nca.fit(data_feature, data_label)
        print("Cost sensitive - Neighbor Component Analysis Done!")
        return nca
    
    def WS_SMOTE(self, data_feature, data_label, nca):
        data_feature = nca.transform(data_feature)
        feature_smote_new, label_smote_new = np.empty((0,data_feature.shape[1])), np.empty((0,1))
        while sum(label_smote_new)+sum(data_label) < self.imbalanced_threshold*(len(data_label)-sum(data_label)):
            #利用SMOTE方法生成少量样本
            #smo = BorderlineSMOTE(sampling_strategy=1, random_state=42, kind="borderline-1")
            smo = SMOTE(sampling_strategy=1, random_state=42)
            feature_smote, _ = smo.fit_resample(data_feature, data_label)
            feature_smote = feature_smote[data_feature.shape[0]:feature_smote.shape[0]]    
            
            #利用图半监督学习确定生成样本属于少数样本的概率
            low_LU = np.append(data_feature, feature_smote, axis=0)
            pij_mat = pairwise_distances(low_LU, squared = True)
            np.fill_diagonal(pij_mat, np.inf)
            pij_mat = np.exp(0.0 - pij_mat - logsumexp(0.0 - pij_mat, axis = 1)[:, None])
            # 将距离小于阈值的全部设置为0,防止出现浮点溢出
            pij_mat = np.where(pij_mat > 1.0e-5, pij_mat, 0)
            Y_unlabel = np.eye(len(low_LU)-len(data_label)) - pij_mat[len(data_label):len(low_LU),
                              len(data_label):len(low_LU)]
            Y_unlabel = np.linalg.pinv(Y_unlabel).dot(pij_mat[len(data_label):len(low_LU),
                                       0:len(data_label)]).dot(data_label)    
            #将属于少数样本概率小于阈值的生成数据剔除
            temp = feature_smote[Y_unlabel[:,0]>=self.pro_threshold,:]
            temp_label = Y_unlabel[Y_unlabel[:,0]>=self.pro_threshold]
            feature_smote_new = np.append(feature_smote_new, temp, axis=0)
            label_smote_new = np.append(label_smote_new, temp_label, axis=0)
      
        #对数据进行维度转化，[samples, time steps, features]
        data_feature = np.reshape(data_feature, (data_feature.shape[0],1,self.Feature_number))
        data_label = np.reshape(data_label, (data_label.shape[0],1,1))
        data_label = np.concatenate((1-data_label, data_label), axis=2)    
        feature_smote_new = np.reshape(feature_smote_new, (feature_smote_new.shape[0],1,self.Feature_number))
        label_smote_new = np.reshape(label_smote_new, (label_smote_new.shape[0],1,1))
        label_smote_new = np.concatenate((1-label_smote_new, label_smote_new), axis=2)    
        #对生成的数据和原始数据进行合并
        x_train = np.concatenate((data_feature, feature_smote_new), axis=0)
        y_train = np.concatenate((data_label, label_smote_new), axis=0)
        #对训练集进行打乱
        index = [i for i in range(x_train.shape[0])]  
        random.shuffle(index) 
        x_train, y_train = x_train[index], y_train[index]    
        return x_train, y_train
    
    def create_model(self):
        input_shape = (1,self.Feature_number)
        inputs = Input(input_shape)
        #shared_layer is the shared layer, diff_layer belong to every output-head
        shared_layer = Dense(8, activation='elu')(inputs)
        #shared_layer = Dense(6, activation='elu')(shared_layer)
        shared_layer = Dense(4, activation='elu')(shared_layer)
        diff_layer = Dense(4, activation='elu')(shared_layer)
        predictions = Dense(2, activation='softmax')(diff_layer)
        model = Model(inputs=inputs, outputs=predictions)
        return model
    
    def train_model(self, model, diff_layer_num, x_train, y_train):
        model.load_weights('shared_weight.h5', by_name=True)
        for layer in model.layers[:-diff_layer_num]:
            #print(layer.trainable)
            layer.trainable = False
        
        early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0.001, patience=50, verbose=2)
        model.compile(loss='categorical_crossentropy', 
                      optimizer='adam', metrics=['accuracy'])  #categorical_crossentropy
        model.fit(x=x_train, y=y_train,  #设置输入的特征值
                  validation_split=0.2,         #设置相应标签值，进行训练集验证集的划分 
                  epochs=200, batch_size=self.BatchSize, verbose=0,
                  callbacks=[early_stopping])
        return model
    
    def class_score(self, y_test, test_prediction):
        TP, TN, FP, FN = 0, 0, 0, 0
        n = len(y_test)
        for i in range(n):
          if y_test[i,] == 1:
            if test_prediction[i,] == 1:
              TP += 1
            else:
              FN += 1
          else:
            if test_prediction[i,] == 0:
              TN += 1
            else:
              FP += 1
        #防止TP，TN为0，影响后续计算
        TP, TN= max(TP,1), max(TN,1)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fscore = 2 * precision * recall / (precision + recall)
        TN_rate = TN / (TN + FP)
        gmean = math.sqrt(TN_rate*recall)
        GF = fscore + gmean
        return TP, TN, FP, FN, fscore, gmean, GF
    
    def get_ensemble_prediction(self, bootstrap_test, train_prediction):
        weight = train_prediction[:,6]/sum(train_prediction[:,6])
        bootstrap_test = np.reshape(weight,(weight.shape[0],1)) * bootstrap_test[:,:,1]
        test_prediction = bootstrap_test.sum(0)
        for i in range(len(test_prediction)):
            if (test_prediction[i,] > 0.5):
                test_prediction[i,] = 1
            else:
                test_prediction[i,] = 0
        return test_prediction
    
    
    def classifier_result_save(self, score, saveFile_name):
        score_df = pd.DataFrame(score)
        score_df.columns = ['TP', 'TN', 'FP', 'FN', 'F-measure', 'G-mean', 'F+G']
        score_df.index = ['CV1', 'CV2', 'CV3', 'CV4', 'CV5']
        writer = pd.ExcelWriter(saveFile_name)
        score_df.to_excel(writer,'Page 1',float_format='%.3f') # float_format 控制精度
        writer.save()
        
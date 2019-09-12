# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:52:15 2019

@author: dell
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset5=pd.read_csv('BankCustomer.csv')

x=dataset5.iloc[:,3:13]

y=dataset5.iloc[:,13]

# Now very important point is that colume no 4(Geography)and 6(Gender) both are categorical data so we cant 
#use as it is so forst convert into dummy variable with the help of Pandas or Machine learning

states=pd.get_dummies(x['Geography'],drop_first=True) #There are three Geography France,Spain,Germany  now drop_first mins delete France and try to find data from two others Geography
                                                     #if Spain=0 ,Germany=0 then France=1 mins this particular colume France is active


gender=pd.get_dummies(x['Gender'],drop_first=True) #same approch of above ,delete one categories minsFemal and with help of Male we can
                                                  #we can find the femal,0=notactive,1=active

#now drop the colume as it is no longer required,mins drop Geography and Gender because allready convert in to dummy variable

x=x.drop(['Geography','Gender'],axis=1)

#now concatenate the remaining dummy variable columns

x=pd.concat([x,states,gender],axis=1)

#split the dataset based on training and test dataset with the help of skitlearn

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/5,random_state=0)

#now there are many feature with different datatype of data so we are going to scalling the data for same datatype foe working

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)                                                 


# now going to Part 2 - now going to make the ANNI

#importing the keras libraries and packages

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
 # Initialising the ANN
 
classifier = Sequential()
  
 #Adding the input layer and the first hidden layer
 
classifier.add(Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform'))
classifier.add(Dropout(p=0.1))
 
#Adding second hidden layer

classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform' )) 
classifier.add(Dropout(p=0.1))
#Adding output layer

classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))

# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

# fiting the ANN to the Training set

classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)

# Part 3 Making the prediction and evaluating the model

# Predicting the Test set results

y_pred = classifier.predict(x_test)
y_pred =  (y_pred > 0.5)

# Making the confusion maatrics
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
accuracy=accuracy_score(y_test,y_pred)



# Predicting a simple new observation
# Predict if the customer with the following information will leave the bank:
#Geogrsphy:France
#Creadit score: 600
#Gender:Male
#Age:40
#Tenure:3
#Balence:40000
#Number of Product:2
#Has creadit cards:yes
#Is active member:yes
#Stimed salary:50000

new_prediction = classifier.predict(sc.transform(np.array([[600,40,3,40000,2,1,1,50000,0,0,1]])))
new_prediction =  (new_prediction > 0.5)


#Part 4 Evaluating,Improving and Tuning the ANI

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
     # Initialising the ANN
    classifier = Sequential()
    classifier.add(Dense(activation='relu', input_dim=11, units=6, kernel_initializer='uniform'))
    classifier.add(Dense(activation='relu', units=6, kernel_initializer='uniform' )) 
    classifier.add(Dense(activation='sigmoid', units=1, kernel_initializer='uniform'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier,batch_size = 10,nb_epoch = 100)
accuracies = cross_val_score(estimator = classifier,x = x_train, y=y_train,cv = 10,n_jobs = -1)

    
    


    
    
    
 
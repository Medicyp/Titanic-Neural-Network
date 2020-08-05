import pandas as pd
import pandas as pd
#Import train data and remove columns which are not useful
dftrain = pd.read_csv('C:/Users/Djinn/Documents/Word/Divers/Geek/Machine learning/Projet 11 - Kaggle Titanic/train.csv')
​
dftrain = dftrain.drop(['PassengerId','Name','Ticket','SibSp','Parch','Cabin','Embarked'],1)
​
print(dftrain.head())
   Survived  Pclass     Sex   Age     Fare
0         0       3    male  22.0   7.2500
1         1       1  female  38.0  71.2833
2         1       3  female  26.0   7.9250
3         1       1  female  35.0  53.1000
4         0       3    male  35.0   8.0500
#Import test data and remove columns which are not useful
dftest = pd.read_csv('C:/Users/Djinn/Documents/Word/Divers/Geek/Machine learning/Projet 11 - Kaggle Titanic/test.csv')
​
initial_dftest = dftest
​
dftest = dftest.drop(['PassengerId','Name','Ticket','SibSp','Parch','Cabin','Embarked'],1)
​
print(dftest.head())
   Pclass     Sex   Age     Fare
0       3    male  34.5   7.8292
1       3  female  47.0   7.0000
2       2    male  62.0   9.6875
3       3    male  27.0   8.6625
4       3  female  22.0  12.2875
#Replacing NaN by the average value for Age
dftrain['Age'].fillna(dftrain['Age'].mean(), inplace=True)
#Normalize the train data set
Number_of_Class = dftrain['Pclass'].max()
Max_Age = dftrain['Age'].max()
Max_Fare = dftrain['Fare'].max()
​
#Replacing NaN by the average value for Age
dftrain['Age'].fillna(dftrain['Age'].mean(), inplace=True)
​
dftrain['Pclass']=dftrain['Pclass']/Number_of_Class
dftrain['Sex'].replace('male',1,inplace=True)
dftrain['Sex'].replace('female',0,inplace=True)
dftrain['Age']=dftrain['Age']/Max_Age
dftrain['Fare']=dftrain['Fare']/Max_Fare
print(dftrain.head())
   Survived    Pclass  Sex     Age      Fare
0         0  1.000000    1  0.2750  0.014151
1         1  0.333333    0  0.4750  0.139136
2         1  1.000000    0  0.3250  0.015469
3         1  0.333333    0  0.4375  0.103644
4         0  1.000000    1  0.4375  0.015713
#Normalize the test data set
​
#Replacing NaN by the average value for Age
dftest['Age'].fillna(dftest['Age'].mean(), inplace=True)
​
dftest['Pclass']=dftest['Pclass']/Number_of_Class
dftest['Sex'].replace('male',1,inplace=True)
dftest['Sex'].replace('female',0,inplace=True)
dftest['Age']=dftest['Age']/Max_Age
dftest['Fare']=dftest['Fare']/Max_Fare
print(dftest.head())
     Pclass  Sex      Age      Fare
0  1.000000    1  0.43125  0.015282
1  1.000000    0  0.58750  0.013663
2  0.666667    1  0.77500  0.018909
3  1.000000    1  0.33750  0.016908
4  1.000000    0  0.27500  0.023984
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
​
#columns = ['Pclass', 'SexInt', 'EmbarkedInt', 'Age', 'TitleInt','Fare', 'Friends', 'Male_Friends_Survived', 'Male_Friends_NotSurvived', 'Female_Friends_Survived', 'Female_Friends_NotSurvived','MotherOnBoard', 'MotherSurvived', 'ChildOnBoard', 'ChildSurvived', 'ChildNotSurvived']
x_train, x_test, y_train, y_test = train_test_split(dftrain.drop(['Survived'],1), dftrain['Survived'], test_size=0.2, random_state=123)
​
print('dftrain:\n', dftrain.head())
print('x_train:\n', x_train)
print('x_test:\n', x_test)
print('y_train:\n', y_train)
print('y_test:\n', y_test)
dftrain:
    Survived    Pclass  Sex     Age      Fare
0         0  1.000000    1  0.2750  0.014151
1         1  0.333333    0  0.4750  0.139136
2         1  1.000000    0  0.3250  0.015469
3         1  0.333333    0  0.4375  0.103644
4         0  1.000000    1  0.4375  0.015713
x_train:
        Pclass  Sex      Age      Fare
329  0.333333    0  0.20000  0.113168
749  1.000000    1  0.38750  0.015127
203  1.000000    1  0.56875  0.014102
421  1.000000    1  0.26250  0.015094
97   0.333333    1  0.28750  0.123667
..        ...  ...      ...       ...
98   0.666667    0  0.42500  0.044893
322  0.666667    0  0.37500  0.024106
382  1.000000    1  0.40000  0.015469
365  1.000000    1  0.37500  0.014151
510  1.000000    1  0.36250  0.015127

[712 rows x 4 columns]
x_test:
        Pclass  Sex       Age      Fare
172  1.000000    0  0.012500  0.021731
524  1.000000    1  0.371239  0.014110
452  0.333333    1  0.375000  0.054164
170  0.333333    1  0.762500  0.065388
620  1.000000    1  0.337500  0.028213
..        ...  ...       ...       ...
388  1.000000    1  0.371239  0.015086
338  1.000000    1  0.562500  0.015713
827  0.666667    1  0.012500  0.072227
773  1.000000    1  0.371239  0.014102
221  0.666667    1  0.337500  0.025374

[179 rows x 4 columns]
y_train:
 329    1
749    0
203    0
421    0
97     1
      ..
98     1
322    1
382    0
365    0
510    1
Name: Survived, Length: 712, dtype: int64
y_test:
 172    1
524    0
452    0
170    0
620    0
      ..
388    0
338    1
827    1
773    0
221    0
Name: Survived, Length: 179, dtype: int64
sklearn.preprocessing import StandardScaler
#Build the Neural network
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.regularizers import l2, l1
from sklearn.preprocessing import StandardScaler
Using TensorFlow backend.
#Create the architecture
model = Sequential()
x_test_scaled = stdScaler.transform(x_test)
#Scale
stdScaler = StandardScaler()
x_train_scaled = stdScaler.fit_transform(x_train)
x_test_scaled = stdScaler.transform(x_test)
4
#Add the layers
model.add(Dense(1600, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
#Compile the model
model.compile(loss='binary_crossentropy',optimizer='rmsprop')
hist=
#Train the model
hist=model.fit(x_train_scaled,y_train,batch_size=32,epochs=20)
Epoch 1/20
712/712 [==============================] - 0s 231us/step - loss: 0.5401
Epoch 2/20
712/712 [==============================] - 0s 85us/step - loss: 0.4670
Epoch 3/20
712/712 [==============================] - 0s 71us/step - loss: 0.4504
Epoch 4/20
712/712 [==============================] - 0s 66us/step - loss: 0.4490
Epoch 5/20
712/712 [==============================] - 0s 64us/step - loss: 0.4461
Epoch 6/20
712/712 [==============================] - 0s 67us/step - loss: 0.4396
Epoch 7/20
712/712 [==============================] - 0s 59us/step - loss: 0.4413
Epoch 8/20
712/712 [==============================] - 0s 57us/step - loss: 0.4376
Epoch 9/20
712/712 [==============================] - 0s 60us/step - loss: 0.4357
Epoch 10/20
712/712 [==============================] - 0s 57us/step - loss: 0.4359
Epoch 11/20
712/712 [==============================] - 0s 66us/step - loss: 0.4325
Epoch 12/20
712/712 [==============================] - 0s 59us/step - loss: 0.4331
Epoch 13/20
712/712 [==============================] - 0s 59us/step - loss: 0.4289
Epoch 14/20
712/712 [==============================] - 0s 60us/step - loss: 0.4317
Epoch 15/20
712/712 [==============================] - 0s 55us/step - loss: 0.4317
Epoch 16/20
712/712 [==============================] - 0s 64us/step - loss: 0.4303
Epoch 17/20
712/712 [==============================] - 0s 62us/step - loss: 0.4342
Epoch 18/20
712/712 [==============================] - 0s 59us/step - loss: 0.4247
Epoch 19/20
712/712 [==============================] - 0s 59us/step - loss: 0.4312
Epoch 20/20
712/712 [==============================] - 0s 64us/step - loss: 0.4261
x_test_scaled
#Predict surviving score of test part of initial database
result = model.predict(x_test_scaled)
print('Result:', result)
Result: [[0.59511787]
 [0.1089198 ]
 [0.40970206]
 [0.21003062]
 [0.12655655]
 [0.08971158]
 [0.88987345]
 [0.92701125]
 [0.46909225]
 [0.4607056 ]
 [0.40595883]
 [0.5075407 ]
 [0.9829359 ]
 [0.13479221]
 [0.9839836 ]
 [0.5675697 ]
 [0.97159076]
 [0.40980944]
 [0.29161346]
 [0.11553776]
 [0.44739872]
 [0.9291141 ]
 [0.09239459]
 [0.3144911 ]
 [0.30012047]
 [0.13584104]
 [0.14699644]
 [0.976154  ]
 [0.16681176]
 [0.10926142]
 [0.5175751 ]
 [0.8914479 ]
 [0.10200232]
 [0.13765028]
 [0.21564937]
 [0.2560476 ]
 [0.05309242]
 [0.03311101]
 [0.93300915]
 [0.34148473]
 [0.9699551 ]
 [0.11811903]
 [0.4799857 ]
 [0.25661   ]
 [0.19480515]
 [0.1355916 ]
 [0.9900732 ]
 [0.34514928]
 [0.1594215 ]
 [0.9223032 ]
 [0.49391827]
 [0.4507479 ]
 [0.1560443 ]
 [0.08046168]
 [0.14511651]
 [0.10891706]
 [0.17254883]
 [0.17510337]
 [0.40223485]
 [0.17297736]
 [0.9911567 ]
 [0.9049977 ]
 [0.42994642]
 [0.9812496 ]
 [0.13685623]
 [0.09013742]
 [0.07876697]
 [0.0761289 ]
 [0.4133999 ]
 [0.92911065]
 [0.9855499 ]
 [0.03478682]
 [0.41304013]
 [0.23173815]
 [0.15281072]
 [0.11841324]
 [0.05997992]
 [0.1528028 ]
 [0.98731446]
 [0.9697355 ]
 [0.4854918 ]
 [0.12056968]
 [0.24849236]
 [0.1089198 ]
 [0.13702393]
 [0.10393596]
 [0.5774574 ]
 [0.09161136]
 [0.41434085]
 [0.13231489]
 [0.49173254]
 [0.11180532]
 [0.07169396]
 [0.31860906]
 [0.36613065]
 [0.1668328 ]
 [0.98592806]
 [0.97640836]
 [0.09160554]
 [0.92130506]
 [0.31223893]
 [0.30910414]
 [0.29729533]
 [0.50539   ]
 [0.10926142]
 [0.4021571 ]
 [0.12810856]
 [0.11541197]
 [0.08046168]
 [0.909363  ]
 [0.595716  ]
 [0.3997608 ]
 [0.96303236]
 [0.09717375]
 [0.12806499]
 [0.09243178]
 [0.2049551 ]
 [0.07838655]
 [0.17511085]
 [0.49900213]
 [0.16681176]
 [0.05042189]
 [0.31858188]
 [0.10973364]
 [0.12514386]
 [0.1850616 ]
 [0.29359642]
 [0.4135822 ]
 [0.4932034 ]
 [0.15940058]
 [0.9884676 ]
 [0.1132817 ]
 [0.13248008]
 [0.92525756]
 [0.9852408 ]
 [0.1751068 ]
 [0.93467534]
 [0.44264385]
 [0.10946137]
 [0.88614357]
 [0.01966476]
 [0.39858225]
 [0.18467706]
 [0.6196433 ]
 [0.5114609 ]
 [0.11929557]
 [0.07795271]
 [0.12598294]
 [0.12380496]
 [0.98861444]
 [0.302961  ]
 [0.19170612]
 [0.5105918 ]
 [0.19525689]
 [0.98887897]
 [0.1089198 ]
 [0.97599924]
 [0.540497  ]
 [0.87892663]
 [0.98446774]
 [0.88913345]
 [0.31596464]
 [0.43807662]
 [0.96373606]
 [0.39336938]
 [0.99071634]
 [0.82746536]
 [0.15940058]
 [0.4931858 ]
 [0.06434831]
 [0.14174998]
 [0.4143079 ]
 [0.43843666]
 [0.12587643]
 [0.1092476 ]
 [0.04515487]
 [0.5669103 ]
 [0.10891707]
 [0.1218047 ]]
#Check how well the model is finding the score on the test part of the initial database
rightnum = 0
for i in range(0, result.shape[0]):
    if result[i] >= 0.5:
        result[i] = 1
    else:
        result[i] = 0
    if result[i] == y_test.iloc[i]:
        rightnum += 1
print('Prediction accuracy:', rightnum/result.shape[0])
Prediction accuracy: 0.8491620111731844
titanic_predict_NN.csv
import numpy as np
​
x_dftest_scaled = stdScaler.transform(dftest)
​
predict_NN = model.predict(x_dftest_scaled)
​
np.nan_to_num(predict_NN,0)
​
for i in range(0, predict_NN.shape[0]):
    if predict_NN[i] >= 0.5:
        predict_NN[i] = 1
    else:
        predict_NN[i] = 0
​
predict_NN = predict_NN.reshape((predict_NN.shape[0]))
predict_NN = predict_NN.astype('int')
​
submission = pd.DataFrame({
        'PassengerId': initial_dftest['PassengerId'],
        'Survived': predict_NN
    })
​
print(submission)
submission.to_csv("titanic_predict_NN.csv", index=False)
     PassengerId  Survived
0            892         0
1            893         0
2            894         0
3            895         0
4            896         0
..           ...       ...
413         1305         0
414         1306         1
415         1307         0
416         1308         0
417         1309         0

[418 rows x 2 columns]

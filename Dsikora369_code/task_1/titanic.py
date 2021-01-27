# -*- coding: utf-8 -*-
"""Titanic.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11X_XykBliOu0vOzxkhNBvPBXDekZWxgu

## Titanic competition Sikorskyi ##

**Downloading datasets from kaggle**

Using Google Colabs functions download Kaggle token.
"""

import pandas as pd
import numpy as np
import re
from google.colab import files
files.upload() #here you will download kaggle.json

"""Set permission to before downloading Titanic dataset."""

!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json #set permission

"""Downloading dataset.

"""

!kaggle competitions download -c titanic

"""Getting our data.

"""

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train

"""**Data transformation**

Befor creating model, it is necessary to analyze, then create function to clean data.

Sibsp and Parch refer to family, so it is logical to replace them with one column and create 'Alone' column for people without family members.
"""

train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index = False).mean()

train['Alone'] = 0
train.loc[train['FamilySize'] == 1, 'Alone'] = 1
train[['Alone', 'Survived']].groupby(['Alone'], as_index=False).mean()

"""'Fare' has NaNs, so they will be replaced to mean. Also, for future model it is better to replace 'Fare' with its categorical representation. """

train['Fare'] = train['Fare'].fillna(train['Fare'].median())
train['CategoricalFare'] = pd.qcut(train['Fare'], 5)
train[['CategoricalFare', 'Survived']].groupby(['CategoricalFare'], as_index=False).mean()

"""Also, it is very useful to notice whether passanger had a cabin or not and which cabin.

"""

def get_cabin(cabin):
  for i in str(cabin):
    if i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Z']:
      return i
train['Cabin'] = train['Cabin'].fillna('Z')
train['BL'] = train['Cabin'].apply(get_cabin)
train[['BL', 'Survived']].groupby(['BL'], as_index=False).mean()

"""In 'Age' we will replace NaNs with mean random and transform to categorical."""

age_avg        = train['Age'].mean()
age_std        = train['Age'].std()    
age_null_count = train['Age'].isnull().sum()
    
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size = age_null_count)
train['Age'][np.isnan(train['Age'])] = age_null_random_list
train['Age'] = train['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)

train[['CategoricalAge', 'Survived']].groupby(['CategoricalAge'], as_index=False).mean()

"""We can find out different titles of passangers."""

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

train['Title'] = train['Name'].apply(get_title)

pd.crosstab(train['Title'], train['Sex'])

"""So we can transform titles to categorical."""

train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
    'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

"""Column 'Embarked' has NaNs that will be replced with median."""

train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

"""Adding Ageclass feature.

"""

train['Age*Class'] = train.Age * train.Pclass 
train

"""Dropping useless columns."""

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp',
                         'Parch', 'FamilySize']

train = train.drop(drop_elements, axis=1)
train

"""Creating function for data preprocessing and dividing into training and validation data."""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
def get_title(name):
        pattern = ' ([A-Za-z]+)\.'
        title_search = re.search(pattern, name)
        # If the title exists, extract and return it.
        if title_search:
            return title_search.group(1)
        return ""
def get_cabin(cabin):
  for i in str(cabin):
    if i in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'Z']:
      return i
def load_data(name):
        dataset = pd.read_csv(name)
        # columns combination
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

        # replace value
        dataset['Alone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'Alone'] = 1

        # fill Nan with mode
        dataset['Embarked'] = dataset['Embarked'].fillna(dataset['Embarked'].mode()[0])

        # fill Nan with median
        dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
        # binning with qcut
        dataset['Fare'] = pd.qcut(dataset['Fare'], 4)

        # fill Nan with mean
        age_avg = dataset['Age'].mean()
        age_std = dataset['Age'].std()
        age_null_count = dataset['Age'].isnull().sum()
        rng = np.random.RandomState(42)
        age_null_random_list = rng.uniform(age_avg - age_std, age_avg + age_std, size=age_null_count)
        dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

        # binning with cut
        dataset['Age'] = pd.cut(dataset['Age'], 5)

        # apply regex
        dataset['Title'] = dataset['Name'].apply(get_title)
        # replace
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don',
                                                               'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'],
                                                               'Rare')
        # replace
        dataset['Title'] = dataset['Title'].replace(['Mlle', 'Ms'], 'Miss')
        # replace
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        # fill nans
        dataset['Title'] = dataset['Title'].fillna(0)
        #fill nans
        dataset['Cabin'] = dataset['Cabin'].fillna('Z')
        #apply replacement 
        dataset['BL'] = str(dataset['Cabin'].apply(get_cabin))
        


        # drop columns
        drop_elements = ['PassengerId', 'Name', 'Ticket', 'SibSp',
                         'Parch', 'FamilySize', 'Cabin']

        dataset = dataset.drop(drop_elements, axis=1)

        # encode labels
        le = LabelEncoder()

        le.fit(dataset['Sex'])
        dataset['Sex'] = le.transform(dataset['Sex'])
        
        le.fit(dataset['Title'])
        dataset['Title'] = le.transform(dataset['Title'])

        le.fit(dataset['Embarked'].values)
        dataset['Embarked'] = le.transform(dataset['Embarked'].values)

        le.fit(dataset['BL'])
        dataset['BL'] = le.transform(dataset['BL'])

        le.fit(dataset['Fare'])
        dataset['Fare'] = le.transform(dataset['Fare'])

        le.fit(dataset['Age'])
        dataset['Age'] = le.transform(dataset['Age'])
        #creating ageclass feature
        dataset['Age*Class'] = dataset.Age * dataset.Pclass
        le.fit(dataset['Age*Class'])
        dataset['Age*Class'] = le.transform(dataset['Age*Class'])

        return dataset
dataset = load_data("train.csv")
train, val = train_test_split(dataset, test_size=0.1)
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Alone', 'Title', 'BL', 'Age*Class']
training_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(train[features].values, tf.float32),
            tf.cast(train['Survived'].values, tf.int32)
        )
    )
)
validation_dataset = (
    tf.data.Dataset.from_tensor_slices(
        (
            tf.cast(val[features].values, tf.float32),
            tf.cast(val['Survived'].values, tf.int32)
        )
    )
)

"""**Creating model**

"""

import tensorflow as tf
import math
from tensorflow.keras.layers import Dense, Dropout
model = tf.keras.Sequential([
              Dense(12, activation = tf.nn.leaky_relu, input_shape = [9]), 
              Dropout(0.2),
              Dense(5, activation = tf.nn.leaky_relu),
              Dense(2, activation = tf.nn.softmax)
])
opt = tf.keras.optimizers.Adam(
    learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True,
    name='Adam'
)
model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

num_train_examples = [i for i,_ in enumerate(training_dataset)][-1] + 1
num_val_examples = [i for i,_ in enumerate(validation_dataset)][-1] + 1
BATCH_SIZE = 256
train_dataset = training_dataset.cache().repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
val_dataset = validation_dataset.cache().batch(BATCH_SIZE)
model.fit(train_dataset, validation_data = val_dataset, epochs=1000, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))

"""Predicting and saving."""

model.save('/content/model.h5')
test_ids = pd.read_csv("test.csv")["PassengerId"]
test_dataset = load_data("test.csv")
Y_pred = model.predict(test_dataset)
Y_pred = (Y_pred[:, 1]>=0.8).astype(int) 

submission = pd.DataFrame({
        "PassengerId": test_ids,
        "Survived": Y_pred
    })
submission.to_csv('/content/titanic.csv', index=False)
print('Done!')
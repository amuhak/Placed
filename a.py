import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
df = pd.read_csv("collegePlace.csv",header=None)
names = ["Age","Gender","Stream","Internships","CGPA","Hostel","HistoryOfBacklogs","PlacedOrNot"]
df.columns = names
print(df.head())
print(df.info())
print(df.describe())
label_encode = {"Stream": {"Electronics And Communication":0, "Computer Science":1, "Information Technology":2, "Civil":3, "Electrical":4, "Mechanical":5}}
df.replace(label_encode,inplace=True)
x_values = df[["Age","Gender","Stream","Internships","CGPA","Hostel","HistoryOfBacklogs"]]
print(x_values.head())
#standardise = StandardScaler() 
#x_values = standardise.fit_transform(x_values)
x_values_df = pd.DataFrame(x_values)

model = Sequential()
model.add(Dense(64,input_dim=7,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(1024,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(2,activation='softmax'))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

print(df['PlacedOrNot'].value_counts())
print(df['PlacedOrNot'].value_counts())
y_values = df['PlacedOrNot']
y_values = to_categorical(y_values)
print(y_values)
model.fit(x_values,y_values,epochs=100,shuffle=True, batch_size=1)
#may take a long time

Age=22
Gender=1
Stream=3
Internships=0
CGPA=2
Hostel=0
HistoryOfBacklogs=1
print(Age,Gender,Stream,Internships,CGPA,Hostel,HistoryOfBacklogs)
in1=[[Age,Gender,Stream,Internships,CGPA,Hostel,HistoryOfBacklogs]]
model.predict(in1)
#first value is false second true
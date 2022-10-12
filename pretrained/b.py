#pretrained modal
import tensorflow as tf
import sklearn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
import os

model = tf.keras.models.load_model('saved_model/my_model')

model.summary()


while True:
    print("ENTER ONLY INT VALUES")
    Age=int(input("enter age (int value) "))
    Gender=int(input("enter gender (int value) ")) #0==male,1==female
    Stream=int(input("enter stream (int vaue) "))
    Internships=int(input("enter internships (int vaue) "))
    CGPA=int(input("enter CGPA(int vaue) "))
    Hostel=int(input("enter Hostel(int vaue) "))
    HistoryOfBacklogs=int(input("enter if HistoryOfBacklogs(int vaue) "))
    print("Age,Gender,Stream,Internships,CGPA,Hostel,HistoryOfBacklogs")
    print(Age,Gender,Stream,Internships,CGPA,Hostel,HistoryOfBacklogs)
    in1=[[Age,Gender,Stream,Internships,CGPA,Hostel,HistoryOfBacklogs]]
    temp=model.predict(in1)
    temp1=temp[0]
    if(temp1[0]>temp1[1]):
        print("""Won't get placed \nprobability= """,temp1[0])
    else:
        print("""Will get placed \nprobability= """,temp1[1])
    #first value is false second true

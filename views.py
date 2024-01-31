# from django.contrib.auth import authenticate, login
from django.shortcuts import render, HttpResponse, redirect
from django.db import models
from .models import userdetails, cropdetails, pestdetect, rentingtool, expertadvice, weatherreport
from django.contrib.auth.models import User, auth
from django.contrib import messages
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from bs4 import BeautifulSoup
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import glob as gb
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense ,Flatten ,Conv2D ,MaxPooling2D ,Dropout ,BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.callbacks import EarlyStopping ,ReduceLROnPlateau , ModelCheckpoint
from keras.applications.mobilenet import MobileNet ,preprocess_input

# Create your views here.
current_datetime = datetime.datetime.now()


def index(request):
    return render(request, "crop_app/index.html", {})


def Login(request):
    if request.method == "POST":
        print("hi")
        email = request.POST["emailid"]
        password = request.POST["password"]
        try:
            user = userdetails.objects.get(emailid=email, password=password)
            request.session['userid'] = user.id
            request.session['uname'] = user.first_name
            return redirect('homepage')
        except:
            pass

    return render(request, "crop_app/Login.html", {})


def register(request):
    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        emailid = request.POST['emailid']
        password = request.POST['password']
        phonenumber = request.POST['phonenumber']
        newuser = userdetails(first_name=first_name, last_name=last_name, emailid=emailid, password=password,
                              phonenumber=phonenumber)
        newuser.save()
        return render(request, "crop_app/index.html", {})
        # return HttpResponse("User added")
    elif request.method == 'GET':
        return render(request, "crop_app/register.html", {})


def homepage(request):
    userid = request.session['userid']

    return render(request, "crop_app/homepage.html", {})


def crop_npk(N, P, K):
    # load the dataset
    crop_data = pd.read_csv('D:/krishi/krishi/krishiproject/datasets/Crop_recommendation.csv')
    print("hi2")

    # extract the features and labels
    X = crop_data[['N', 'P', 'K']].values
    y = crop_data['label'].values

    # create the decision tree model and fit the data
    model = DecisionTreeClassifier()
    model.fit(X, y)

    # input the NPK values for prediction
    N = float(N)
    P = float(P)
    K = float(K)

    # make the prediction using the model
    crop_prediction = model.predict([[N, P, K]])

    # print the predicted crop
    print('The predicted crop for N={}, P={}, K={} is: {}'.format(N, P, K, crop_prediction[0]))
    return crop_prediction[0]


def fertilizer_npk(N, P, K):
    # load the dataset
    fertilizer_data = pd.read_csv('D:/krishi/krishi/krishiproject/datasets/Fertilizer Prediction.csv')

    # extract the features and labels
    X = fertilizer_data[['Nitrogen', 'Phosphorous', 'Potassium']].values
    y = fertilizer_data['Fertilizer Name'].values

    # create the K-Nearest Neighbors model and fit the data
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    # input the NPK values for prediction
    N = float(N)
    P = float(P)
    K = float(K)

    # make the prediction using the model
    fertilizer_prediction = model.predict([[N, P, K]])

    # print the predicted fertilizer
    print('The predicted fertilizer for N={}, P={}, K={} is: {}'.format(N, P, K, fertilizer_prediction[0]))
    return fertilizer_prediction[0]


def croppredict(request):
    uid = request.session['userid']

    if request.method == "POST":
        n = request.POST['nitrogen']
        p = request.POST['phosphorous']
        k = request.POST['potassium']

        cropvalue = crop_npk(n, p, k)
        fertvalue = fertilizer_npk(n, p, k)

        newcrop = cropdetails(userid=uid, nitrogen=n, phosphorous=p, potassium=k, crop_predict=cropvalue,
                              fertilize_predict=fertvalue)
        newcrop.save()

        context = {

            'bdata': "The predicted crop and fertilizer for ",
            'Nitrogen': n,
            'phosphorous': p,
            'potassium': k,
            'con1': "is ",
            'crop_predict': cropvalue,
            'con2': "and ",
            'fertilize_predict': fertvalue,

        }

        return render(request, "crop_app/croppredict.html", context)

    return render(request, "crop_app/croppredict.html")

def predictpest():
    train_dir = 'pest/train/'
    test_dir = 'pest/test/'

    training = tf.keras.preprocessing.image.ImageDataGenerator(
        zca_epsilon=1e-06,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,
        validation_split=0.05
    ).flow_from_directory(train_dir, batch_size=16, target_size=(224, 224), subset="training")

    validing = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input,
        validation_split=0.05
    ).flow_from_directory(train_dir, batch_size=16, target_size=(224, 224), subset='validation', shuffle=True)

    testing = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocess_input,
    ).flow_from_directory(test_dir, batch_size=16, target_size=(224, 224), shuffle=True)

    mobilenet = MobileNet(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.99)
    EarlyStop = EarlyStopping(patience=10, restore_best_weights=True)
    Reduce_LR = ReduceLROnPlateau(monitor='val_acc', verbose=2, factor=0.5, min_lr=0.00001)
    callback = [EarlyStop, Reduce_LR]
    mobilenet.trainable = False
    model = Sequential([
        mobilenet,
        MaxPooling2D(3, 2),
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(1024, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(9, activation='softmax')
    ])
    model.summary()
    model.compile(optimizer='adam', loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    history = model.fit(training, validation_data=validing, epochs=20, batch_size=16,
                        steps_per_epoch=len(training) // 16, validation_steps=len(validing) // 8,
                        callbacks=callback, verbose=2)

def predictimg(filename):
    def load_image(filename):
        # load the image
        img = load_img(filename, grayscale=True, target_size=(28, 28))
        # convert to array
        img = img_to_array(img)
        # reshape into a single sample with 1 channel
        img = img.reshape(1, 28, 28, 1)
        # prepare pixel data
        img = img.astype('float32')
        img = img / 255.0
        return img

    pic = []
    filepath = "D:/krishi/krishi/krishiproject/pesttestimages/"
    filename =filename

    imgpath = filepath + filename
    print(imgpath)
    img = cv2.imread(str(imgpath))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img / 255
    # label = to_categorical(0, num_classes=2)
    pic.append(img)
    pic1 = np.array(pic)
    pic1
    model = load_model('myModel.h5')
    a = model.predict(pic1)
    a
    print(a.argmax())
    return a.argmax()

def pestdetect(request):
    if request.method == "POST":
        print("hicr")
        filename= request.POST["myfile"]
        pestname={0:"Pest Detected is: Aphids",1:'Pest Detected is: Armyworm',2:'Pest Detected is: Beetle',3:'Pest Detected is: Bollworm',4:'Pest Detected is: Grasshopper',5:'Pest Detected is: Mites',6:'Pest Detected is: Mosquito',7:'Pest Detected is: Awfly',8:'Pest Detected is: Stem_Borer'}
        res=predictimg(filename)
        pestresult=pestname[res]
        print(pestresult)
        context={
            "result":pestresult

        }
        return render(request, "crop_app/pestdetect.html",context)
    return render(request, "crop_app/pestdetect.html")


def cityweather(city):
    # Specify the URL of the website providing weather details
    url = f'https://www.example.com/weather/{city}'

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        print("Weather Report")
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the specific elements containing weather information
        temperature_element = soup.select('#temperature')[0]
        humidity_element = soup.select('#humidity')[0]
        description_element = soup.select('#description')[0]

        # Extract the text from the elements
        temperature = temperature_element.get_text().strip()
        humidity = humidity_element.get_text().strip()
        description = description_element.get_text().strip()

        # Return a formatted weather report
        weather_report = f"Weather report for {city}:\nTemperature: {temperature}\nHumidity: {humidity}\nDescription: {description}"
        return weather_report

    else:
        # If the request was not successful, print an error message
        print('Error retrieving weather information. Status Code:', response.status_code)


def weatherpredict(request):
    uid = request.session['userid']
    if request.method == "POST":
        print("Hi")
        locn = request.POST['location']
        print(locn)
        data = locn.lower()
        if data=='nelmangala':
            res='29'
        elif data=='kammasandra':
            res='29'
        elif data=='bangalore':
            res='30'
        elif data=='delhi':
              res=  '41'
        elif data=='chennai':
            res='34'
        else:
            res="Enter city name correctly"
        # locweather = cityweather(locn)
        print(data)
        newweather = weatherreport(userid=uid, locationname=locn, temp=res, todaydate=current_datetime)
        newweather.save()

        context = {

            'bdata': "The predicted weather for  ",
            'locn': locn,
            'con1': "is ",
            'locweather': res,
            'con2': "°C",

        }

        return render(request, "crop_app/weatherpredict.html", context)
    return render(request, "crop_app/weatherpredict.html")


def toolscost(atool):
    if atool == "Axe":
        print("100")
        return 100
    elif atool == "Rake":
        print("200")
        return 200
    elif atool == "Shovel":
        print("300")
        return 300
    elif atool == "Fork":
        print("400")
        return 400
    elif atool == "Saw":
        print("500")
        return 500
    elif atool == "Shears":
        print("600")
        return 600
    elif atool == "Wheelbarrow":
        print("700")
        return 700
    elif atool == "Land Mover":
        print("800")
        return 800
    elif atool == "Seeder":
        print("900")
        return 190
    elif atool == "Machete":
        print("12200")
        return 12200
    elif atool == "Tractor":
        print("2000")
        return 2000
    elif atool == "Can":
        print("1000")
        return 1000


def rentingtools(request):
    uid = request.session['userid']
    if request.method == "POST":
        atool = request.POST["tools"]
        print(atool)
        tcost = toolscost(atool)

        newweather = rentingtool(userid=uid, toolname=atool, toolcost=tcost, todaydate=current_datetime)
        newweather.save()

        context = {

            'bdata': "The cost of rented tool   ",
            'atool': atool,
            'con1': "is ",
            'con2': "Rs.",
            'tcost': tcost,

        }

        return render(request, "crop_app/rentingtools.html", context)

    return render(request, "crop_app/rentingtools.html")


def advice(eadvice):
    if eadvice == "Q1":
        a = (
            "Organic farming, subsistence farming, commercial farming are popular farming methods used in India. However, depending on geographical conditions, production demand, level of technology and labour, farming can be based on ley farming, horticulture, agroforestry, etc.")
        return a
    elif eadvice == "Q2":
        a = (
            "Mixed farming, shifting agriculture, intensive farming, crop rotation, plantation agriculture, arable farming are few popular types of agriculture practices")
        return a
    elif eadvice == "Q3":
        a = ("There are three season crops such as Zaid, Rabi, and Kharif in India.")
        return a
    elif eadvice == "Q4":
        a = ("They are summer season crops, grown for short periods between March to June.")
        return a
    elif eadvice == "Q5":
        a = ("The Kharif season in India starts in June and ends in October. ")
        return a
    elif eadvice == "Q6":
        a = (
            " Rabi crops are grown in winter between October to November. Barley, Oats, Wheat, Pulses are few examples of Rabi crops.")
        return a
    elif eadvice == "Q7":
        a = (
            "Modern agriculture involves use of advanced agricultural technology and farming techniques that reduces costs, increases efficiency and crop yield.")
        return a
    elif eadvice == "Q8":
        a = ("Pramod Gautam, Sachin Kale, Harish Dhandev are few top richest farmers in India. ")
        return a
    elif eadvice == "Q9":
        a = (
            "A healthy soil is important as it provides essential nutrients, oxygen, water, and root support to crop producing plants.")
        return a
    elif eadvice == "Q10":
        a = (
            "Organic pesticides are derived from botanical and mineral sources. They contain less chemicals and are less threatening than chemical-based pesticides.")
        return a
    elif eadvice == "Q11":
        a = ("Soil is a non-renewable resource which can take tons of CO2 (carbon) out of the atmosphere.")
        return a
    elif eadvice == "Q12":
        a = (
            " Lack of technology, energy facilities and Irrigation in the rural areas are the reasons for low Indian agricultural GDP.")
        return a


def exprertadvice(request):
    uid = request.session['userid']
    if request.method == "POST":
        quest = request.POST["quest"]
        print(quest)
        qadvice = advice(quest)
        print(qadvice)

        newquest = expertadvice(userid=uid, Questionname=quest, advices=qadvice, todaydate=current_datetime)
        newquest.save()

        context = {
            'bdata': "The advice for the    ",
            'quest': quest,
            'qadvice': qadvice,
        }
        return render(request, "crop_app/exprertadvice.html", context)
    return render(request, "crop_app/exprertadvice.html")

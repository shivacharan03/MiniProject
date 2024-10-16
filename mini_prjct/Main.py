import json
from matplotlib import pyplot as plt
from tkinter import messagebox
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import os
import sys
import numpy as np
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.optimizers import SGD
import keras.callbacks
from PIL import Image, ImageDraw
import tensorflow as tf

root = tkinter.Tk()

root.title("Ship Extraction using Post CNN from High Resolution Optical Remotely Sensed Images")
root.geometry("1200x850")

global filename
global model
global picture_tensor

def cutting(x, y):
    area_study = np.arange(3*80*80).reshape(3, 80, 80)
    for i in range(80):
        for j in range(80):
            area_study[0][i][j] = picture_tensor[0][y+i][x+j]
            area_study[1][i][j] = picture_tensor[1][y+i][x+j]
            area_study[2][i][j] = picture_tensor[2][y+i][x+j]
    area_study = area_study.reshape([-1, 3, 80, 80])
    area_study = area_study.transpose([0,2,3,1])
    area_study = area_study / 255
    sys.stdout.write('\rX:{0} Y:{1}  '.format(x, y))
    return area_study

def not_near(x, y, s, coordinates):
    result = True
    for e in coordinates:
        if x+s > e[0][0] and x-s < e[0][0] and y+s > e[0][1] and y-s < e[0][1]:
            result = False
    return result

def show_ship(x, y, acc, thickness=15):   
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x-th] = -1

    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+i][x+th+80] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y-th][x+i] = -1
        
    for i in range(80):
        for ch in range(3):
            for th in range(thickness):
                picture_tensor[ch][y+th+80][x+i] = -1


def upload():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    

def postCNN():
    global model
    if os.path.exists('ship_model.h5py'):
        model = tf.keras.models.load_model('ship_model.h5py')
        model.summary()
        text.insert(END,"model output can bee seen in black console\n");
    else:
        f = open(r'dataset/shipsnet.json')
        dataset = json.load(f)
        f.close()

        input_data = np.array(dataset['data']).astype('uint8')
        output_data = np.array(dataset['labels']).astype('uint8')

        n_spectrum = 3 # color chanel RGB 
        weight = 80
        height = 80
        X = input_data.reshape([-1, n_spectrum, weight, height])
        pic = X[3]

        y = np_utils.to_categorical(output_data, 2)
        indexes = np.arange(4000)
        np.random.shuffle(indexes)
        X_train = X[indexes].transpose([0,2,3,1])
        y_train = y[indexes]
        X_train = X_train / 255
        np.random.seed(42)

        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(80, 80, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) #40x40
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) #20x20
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) #10x10
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (10, 10), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2))) #5x5
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(2, activation='softmax'))

        sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        # training
        model.fit(X_train, y_train, batch_size=32, epochs=18, validation_split=0.2, shuffle=True, verbose=2)
        model.save("ship_model.h5py1")
        text.insert(END,"model output can bee seen in black console\n");
        
def extractShip():
    global picture_tensor
    test = filedialog.askopenfilename(initialdir="testimage")
    pathlabel.config(text=filename)
    text.insert(END,test+" image loaded\n");

    image = Image.open(r''+test)
    pix = image.load()
    plt.imshow(image)

    n_spectrum = 3
    width = image.size[0]
    height = image.size[1]

    picture_vector = []
    for chanel in range(n_spectrum):
        for y in range(height):
            for x in range(width):
                picture_vector.append(pix[x, y][chanel])

    picture_vector = np.array(picture_vector).astype('uint8')
    picture_tensor = picture_vector.reshape([n_spectrum, height, width]).transpose(1, 2, 0)

    plt.figure(1, figsize = (15, 30))

    plt.subplot(3, 1, 1)
    plt.imshow(picture_tensor)

    plt.show()

    picture_tensor = picture_tensor.transpose(2,0,1)

    step = 30; coordinates = []
    print(int((height-(80-step))/step))
    print(int((width-(80-step))/step) )

    m = 0

    for y in range(int((height-(80-step))/step)):
        if m < 1:
            for x in range(int((width-(80-step))/step) ):
                if m < 1:
                    area = cutting(x*step, y*step)
                    result = model.predict(area)
                    if result[0][1] > 0.91 and not_near(x*step,y*step, 88, coordinates):
                        coordinates.append([[x*step, y*step], result])
                        print(result)
                        plt.imshow(area[0])
                        plt.show()
                        m = m + 1

    print(coordinates)                    

    for e in coordinates:
        show_ship(e[0][0], e[0][1], e[1][0][1])

    picture_tensor = picture_tensor.transpose(1,2,0)
    picture_tensor.shape
    (1777, 2825, 3)
    plt.figure(1, figsize = (15, 30))
    plt.subplot(3,1,1)
    plt.imshow(picture_tensor)
    plt.show()

    

font = ('times', 18, 'bold')
title = Label(root, text='Ship Extraction using Post CNN from High Resolution Optical Remotely Sensed Images')
title.config(bg='wheat', fg='red')  
title.config(font=font)           
title.config(height=3, width=80)       
title.place(x=5,y=5)

font1 = ('times', 14, 'bold')

upload = Button(root, text="Upload Satellite Imagery Dataset", command=upload)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(root)
pathlabel.config(bg='blue', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=400,y=100)

normal = Button(root, text="Run Post CNN Algorithm", command=postCNN)
normal.place(x=50,y=150)
normal.config(font=font1)  

queuebutton = Button(root, text="Upload Test Image & Extract Ship", command=extractShip)
queuebutton.place(x=50,y=200)
queuebutton.config(font=font1)


text=Text(root,height=25,width=80)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=550,y=100)
text.config(font=font1)  

root.mainloop()

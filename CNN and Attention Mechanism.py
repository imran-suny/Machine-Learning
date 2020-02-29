import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPool2D, Activation, Flatten, merge, Lambda, RepeatVector, TimeDistributed, Dense, Dropout, BatchNormalization, Input, Permute, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras import regularizers
print(os.listdir("asl-alphabet"))

train_dir = 'asl-alphabet/asl_alphabet_train/asl_alphabet_train'
test_dir = 'asl-alphabet/asl_alphabet_test/asl_alphabet_test'

def load_unique():
    size_img = 64,64
    images_for_plot = []
    labels_for_plot = []
    for folder in os.listdir(train_dir):
        for file in os.listdir(train_dir + '/' + folder):
            filepath = train_dir + '/' + folder + '/' + file
            image = cv2.imread(filepath)
            final_img = cv2.resize(image, size_img)
            final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
            images_for_plot.append(final_img)
            labels_for_plot.append(folder)
            break
    return images_for_plot, labels_for_plot

images_for_plot, labels_for_plot = load_unique()
print("unique_labels = ", labels_for_plot)
labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25,'space':26,'del':27,'nothing':28}

def load_data():
    images = []
    labels = []
    size = 64,64
    print("LOADING DATA FROM : ",end = "")
    for folder in os.listdir(train_dir):
        print(folder, end = ' | ')
        for image in os.listdir(train_dir + "/" + folder):
            temp_img = cv2.imread(train_dir + '/' + folder + '/' + image)
            temp_img = cv2.resize(temp_img, size)
            images.append(temp_img)
            if folder == 'A':
                labels.append(labels_dict['A'])
            elif folder == 'B':
                labels.append(labels_dict['B'])
            elif folder == 'C':
                labels.append(labels_dict['C'])
            elif folder == 'D':
                labels.append(labels_dict['D'])
            elif folder == 'E':
                labels.append(labels_dict['E'])
            elif folder == 'F':
                labels.append(labels_dict['F'])
            elif folder == 'G':
                labels.append(labels_dict['G'])
            elif folder == 'H':
                labels.append(labels_dict['H'])
            elif folder == 'I':
                labels.append(labels_dict['I'])
            elif folder == 'J':
                labels.append(labels_dict['J'])
            elif folder == 'K':
                labels.append(labels_dict['K'])
            elif folder == 'L':
                labels.append(labels_dict['L'])
            elif folder == 'M':
                labels.append(labels_dict['M'])
            elif folder == 'N':
                labels.append(labels_dict['N'])
            elif folder == 'O':
                labels.append(labels_dict['O'])
            elif folder == 'P':
                labels.append(labels_dict['P'])
            elif folder == 'Q':
                labels.append(labels_dict['Q'])
            elif folder == 'R':
                labels.append(labels_dict['R'])
            elif folder == 'S':
                labels.append(labels_dict['S'])
            elif folder == 'T':
                labels.append(labels_dict['T'])
            elif folder == 'U':
                labels.append(labels_dict['U'])
            elif folder == 'V':
                labels.append(labels_dict['V'])
            elif folder == 'W':
                labels.append(labels_dict['W'])
            elif folder == 'X':
                labels.append(labels_dict['X'])
            elif folder == 'Y':
                labels.append(labels_dict['Y'])
            elif folder == 'Z':
                labels.append(labels_dict['Z'])
            elif folder == 'space':
                labels.append(labels_dict['space'])
            elif folder == 'del':
                labels.append(labels_dict['del'])
            elif folder == 'nothing':
                labels.append(labels_dict['nothing'])
    
    images = np.array(images)
    images = images.astype('float32')/255.0
    
    labels = keras.utils.to_categorical(labels)   #one-hot encoding
    
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size = 0.1)
    
    print()
    print('Loaded', len(X_train),'images for training,','Train data shape =',X_train.shape)
    print('Loaded', len(X_test),'images for testing','Test data shape =',X_test.shape)
    
    return X_train, X_test, Y_train, Y_test
X_train, X_test, Y_train, Y_test = load_data()

## Model for CNN
def build_model():
    
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (64,64,3)))
    model.add(Conv2D(32, kernel_size = 3, padding = 'same', strides = 2, activation = 'relu'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = 3, padding = 'same', strides = 2, activation = 'relu'))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(256, kernel_size = 3, padding = 'same', strides = 2 , activation = 'relu'))
    model.add(MaxPool2D(3))
    
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(29, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])
    
    print("MODEL CREATED")
    model.summary()
    return model

##Model for Attention Mechanism

def build_model_1():
    
    inputs = Input(shape=(64, 64, 3))
    conv1 = Conv2D(32, 3, activation='relu', padding = 'same')(inputs)
    pool1 = MaxPool2D()(conv1)
    
    conv2 = Conv2D(64, 3, activation='relu', padding = 'same')(pool1)
    pool2 = MaxPool2D()(conv2)
    
    conv3 = Conv2D(128, 3, activation='relu', padding = 'same')(pool2)
    pool3 = MaxPool2D()(conv3)
    
    ### Attentinon Mechanism   (class-agnostic attention, class-specific attention)

    
    ### class-agnostic attention
    x = Permute([3, 1, 2])(pool3)
    x = TimeDistributed(Flatten())(x)
    x = Permute([2, 1])(x)
    
    att = Dense(1, activation='tanh')(x)
    
    att=Flatten()(att)
    att=Activation('softmax')(att)
    att=RepeatVector(29)(att)
    att=Permute([2,1])(att)
    
    ### class-specific  attention
    attention_class=Dense(29, activation='softmax')(x)
    multiply = keras.layers.Multiply()([att, attention_class])
    prediction = Lambda(lambda xin: K.sum(xin, axis=1))(multiply)
#    attention_model=Model(inputs=inputs, outputs=representation)
    
    
    model = Model(inputs=inputs, outputs=prediction)
    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])
    
    print("MODEL CREATED")
    model.summary()
    return model

def fit_model():
    history = model.fit(X_train, Y_train, batch_size = 64, epochs = 5, validation_split = 0.1)
    return history
model = build_model_1()
model_history = fit_model() 

# visualize the training and validation loss
hist = model_history
epochs = range(1, len(hist.history['loss']) + 1)

plt.subplots(figsize=(15,6))
plt.subplot(121)
# "bo" is for "blue dot"
plt.plot(epochs, hist.history['loss'], 'bo-')
# b+ is for "blue crosses"
plt.plot(epochs, hist.history['val_loss'], 'ro--')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(122)
plt.plot(epochs, hist.history['acc'], 'bo-')
plt.plot(epochs, hist.history['val_acc'], 'ro--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.show()

if model_history:
    print('Final Accuracy: {:.2f}%'.format(model_history.history['acc'][4] * 100))
    print('Validation Set Accuracy: {:.2f}%'.format(model_history.history['val_acc'][4] * 100)) 

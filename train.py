import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
import random as rn
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

X=[]
Z=[]
IMG_SIZE=150
FLOWER_BARBETONDAISY_DIR='./data/barbeton daisy'
FLOWER_BELL_DIR='./data/bell'
FLOWER_CAPEFLOWER_DIR='./data/cape flower'
FLOWER_FIRELILY_DIR='./data/firelily'
FLOWER_FRITILLARY_DIR='./data/fritillary'
FLOWER_GREATMASTERWORT_DIR='./data/great masterwort'
FLOWER_LOTUS_DIR='./data/lotus'
FLOWER_MARIGOLD_DIR='./data/marigold'
FLOWER_ORCHID_DIR='./data/orchid'
FLOWER_OSTEOSPERMUM_DIR='./data/osteospermum'
FLOWER_PINKYELLOWDAHLIA_DIR='./data/pink-yellow dahlia'
FLOWER_PRIMULA_DIR='./data/primula'
FLOWER_PURPLECONEFLOWER_DIR='./data/purple coneflower'
FLOWER_ROSE_DIR='./data/rose'
FLOWER_SWEETWILLIAM_DIR='./data/sweet william'
FLOWER_THORNAPPLE_DIR='./data/thorn apple'
FLOWER_TRUMPETCREEPER_DIR='./data/trumpet creeper'
FLOWER_WALLFLOWER_DIR='./data/wallflower'
FLOWER_WATERCRESS_DIR='./data/watercress'
FLOWER_WATERLILY_DIR='./data/waterlily'

def assign_label(img,flower_type):
    return flower_type

def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        
        X.append(np.array(img))
        Z.append(str(label))

make_train_data('barbeton daisy',FLOWER_BARBETONDAISY_DIR)
make_train_data('bell',FLOWER_BELL_DIR)
make_train_data('cape flower',FLOWER_CAPEFLOWER_DIR)
make_train_data('firelily',FLOWER_FIRELILY_DIR)
make_train_data('fritillary',FLOWER_FRITILLARY_DIR)
make_train_data('great masterwort',FLOWER_GREATMASTERWORT_DIR)
make_train_data('lotus',FLOWER_LOTUS_DIR)
make_train_data('marigold',FLOWER_MARIGOLD_DIR)
make_train_data('orchid',FLOWER_ORCHID_DIR)
make_train_data('osteospermum',FLOWER_OSTEOSPERMUM_DIR)
make_train_data('pink-yellow dahlia',FLOWER_PINKYELLOWDAHLIA_DIR)
make_train_data('primula',FLOWER_PRIMULA_DIR)
make_train_data('purple coneflower',FLOWER_PURPLECONEFLOWER_DIR)
make_train_data('rose',FLOWER_ROSE_DIR)
make_train_data('sweet william',FLOWER_SWEETWILLIAM_DIR)
make_train_data('thorn apple',FLOWER_THORNAPPLE_DIR)
make_train_data('trumpet creeper',FLOWER_TRUMPETCREEPER_DIR)
make_train_data('wallflower',FLOWER_WALLFLOWER_DIR)
make_train_data('watercress',FLOWER_WATERCRESS_DIR)
make_train_data('waterlily',FLOWER_WATERLILY_DIR)


fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])
        
plt.tight_layout()

le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,20)
X=np.array(X)
X=X/255

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

# Train MOdel

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
 
model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(20, activation = "softmax"))

batch_size=128
epochs=100

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()



# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)

# now storing some properly as well as misclassified indexes'.
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_digits[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==pred_digits[i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


#Hình ảnh đúng của hoa

count = 0
fig, ax = plt.subplots(4, 2)
fig.set_size_inches(15, 15)
for i in range(4):
    for j in range(2):
        ax[i, j].imshow(x_test[prop_class[count]])
        predicted_flower = le.inverse_transform([pred_digits[prop_class[count]]])[0]
        actual_flower = le.inverse_transform(np.argmax(y_test[prop_class[count]]))
        ax[i, j].set_title("Predicted Flower: " + predicted_flower + "\n" + "Actual Flower: " + actual_flower)
        plt.tight_layout()
        count += 1

# Hình ảnh lỗi của hoa

warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count1 = 0
fig, ax = plt.subplots(4, 2)
fig.set_size_inches(15, 15)
for i in range(4):
    for j in range(2):
        ax[i, j].imshow(x_test[mis_class[count1]])
        predicted_flower = le.inverse_transform([pred_digits[mis_class[count1]]])[0]
        actual_flower = le.inverse_transform(np.argmax(y_test[mis_class[count1]]))
        ax[i, j].set_title("Predicted Flower: " + predicted_flower + "\n" + "Actual Flower: " + actual_flower)
        plt.tight_layout()
        count1 += 1

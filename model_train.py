import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
import sys

# Loading Input Images From Folder
def load_images_from_folder(folder):
    train_data=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename),cv2.IMREAD_GRAYSCALE)
        img=~img
        if img is not None:
            ret,thresh=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

            ctrs,ret=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cnt=sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
            w=int(28)
            h=int(28)
            maxi=0
            for c in cnt:
                x,y,w,h=cv2.boundingRect(c)
                maxi=max(w*h,maxi)
                if maxi==w*h:
                    x_max=x
                    y_max=y
                    w_max=w
                    h_max=h
            im_crop= thresh[y_max:y_max+h_max+10, x_max:x_max+w_max+10]
            im_resize = cv2.resize(im_crop,(28,28))
            im_resize=np.reshape(im_resize,(784,1))
            train_data.append(im_resize)
    return train_data

data=[]

# Assigning The Data into Different groups
# Assign '-' = 10
print("importing '-'", end='', flush=True)
data=load_images_from_folder('extracted_images\\-')
len(data)
for i in range(0,len(data)):
    data[i]=np.append(data[i],['10'])
    
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'-\' with', len(data), 'data')

# Assign '+' = 11
print("importing '+'", end='', flush=True)
data11=load_images_from_folder('extracted_images\\+')

for i in range(0,len(data11)):
    data11[i]=np.append(data11[i],['11'])
data=np.concatenate((data,data11))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'+\' with', len(data), 'data')

# Assign '0' = 0
print("importing '0'", end='', flush=True)
data0=load_images_from_folder('extracted_images\\0')

for i in range(0,len(data0)):
    data0[i]=np.append(data0[i],['0'])
data=np.concatenate((data,data0))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'0\' with', len(data), 'data')

# Assign '1' = 1
print("importing '1'", end='', flush=True)
data1=load_images_from_folder('extracted_images\\1')

for i in range(0,len(data1)):
    data1[i]=np.append(data1[i],['1'])
data=np.concatenate((data,data1))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'1\' with', len(data), 'data')

# Assign '2' = 2
print("importing '2'", end='', flush=True)
data2=load_images_from_folder('extracted_images\\2')

for i in range(0,len(data2)):
    data2[i]=np.append(data2[i],['2'])
data=np.concatenate((data,data2))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'2\' with', len(data), 'data')

# Assign '3' = 3
print("importing '3'", end='', flush=True)
data3=load_images_from_folder('extracted_images\\3')

for i in range(0,len(data3)):
    data3[i]=np.append(data3[i],['3'])
data=np.concatenate((data,data3))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'3\' with', len(data), 'data')

# Assign '4' = 4
print("importing '4'", end='', flush=True)
data4=load_images_from_folder('extracted_images\\4')

for i in range(0,len(data4)):
    data4[i]=np.append(data4[i],['4'])
data=np.concatenate((data,data4))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'4\' with', len(data), 'data')

# Assign '5' = 5
print("importing '5'", end='', flush=True)
data5=load_images_from_folder('extracted_images\\5')

for i in range(0,len(data5)):
    data5[i]=np.append(data5[i],['5'])
data=np.concatenate((data,data5))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'5\' with', len(data), 'data')

# Assign '6' = 6
print("importing '6'", end='', flush=True)
data6=load_images_from_folder('extracted_images\\6')

for i in range(0,len(data6)):
    data6[i]=np.append(data6[i],['6'])
data=np.concatenate((data,data6))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'6\' with', len(data), 'data')

# Assign '7' = 7
print("importing '7'", end='', flush=True)
data7=load_images_from_folder('extracted_images\\7')

for i in range(0,len(data7)):
    data7[i]=np.append(data7[i],['7'])
data=np.concatenate((data,data7))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'7\' with', len(data), 'data')

# Assign '8' = 8
print("importing '8'", end='', flush=True)
data8=load_images_from_folder('extracted_images\\8')

for i in range(0,len(data8)):
    data8[i]=np.append(data8[i],['8'])
data=np.concatenate((data,data8))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'8\' with', len(data), 'data')

# Assign '9' = 9
print("importing '9'", end='', flush=True)
data9=load_images_from_folder('extracted_images\\9')

for i in range(0,len(data9)):
    data9[i]=np.append(data9[i],['9'])
data=np.concatenate((data,data9))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'9\' with', len(data), 'data')

# Assign '*' = 12
print("importing '*'", end='', flush=True)
data12=load_images_from_folder('extracted_images\\times')

for i in range(0,len(data12)):
    data12[i]=np.append(data12[i],['12'])
data=np.concatenate((data,data12))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'*\' with', len(data), 'data')

""" # Assign '/' = 13
print("importing '/'", end='', flush=True)
data13=load_images_from_folder('extracted_images\\div')

for i in range(0,len(data13)):
    data13[i]=np.append(data13[i],['13'])
data=np.concatenate((data,data13))
# Clear the line
sys.stdout.write('\r')
sys.stdout.flush()
print('Imported \'/\' with', len(data), 'data') """

# Saving the data
df=pd.DataFrame(data,index=None)
df.to_csv('train_final.csv',index=False)

# Preprocessing

df_train=pd.read_csv('train_final.csv',index_col=False)
labels=df_train[['784']]

df_train.drop(df_train.columns[[784]],axis=1,inplace=True)
df_train.head(10)

# Training The Data
np.random.seed(1212)

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras import optimizers
from tensorflow.keras.utils import to_categorical  # np_utils replaced with to_categorical
from tensorflow.keras import backend as K

labels=np.array(labels)

from tensorflow.keras.utils import to_categorical
cat=to_categorical(labels,num_classes=13)

print(cat[0])

df_train.head(10)

df_train.shape

temp=df_train.to_numpy()

X_train = temp.reshape(temp.shape[0], 28, 28, 1)

temp.shape[0]

X_train.shape

l=[]
for i in range(50621):
    l.append(np.array(df_train[i:i+1]).reshape(1,28,28))

np.random.seed(7)

X_train.shape

# Building the CNN model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28, 28,1), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(15, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(13, activation='softmax'))


# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.models import model_from_json

model.fit(X_train, cat, epochs=10, batch_size=200,shuffle=True,verbose=1)

# serialize model to JSON
model_json = model.to_json()
with open("model_final.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model_final.weights.h5")  # Filename now ends with .weights.h5
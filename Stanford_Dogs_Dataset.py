import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
#import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow_datasets as tfds

# Colab GPU 설정 확인
from tensorflow.python.client import device_lib
device_lib.list_local_devices()

# tensorflow에서 제공하는 StanfordDogs Dataset load하여 classfication 수행
image=tfds.image_classification.StanfordDogs  

# % slicing으로 train, val data 분할
# 수행할 task: classification -> as_supoervised=True 설정해 label을 같이
(train_ds, val_ds, test_ds), metadata=tfds.load("stanford_dogs",
                                      split=['train[:80%]', 'train[80%:]', 'test'],
                                      with_info=True, 
                                      as_supervised=True
                                      )
# dataset의 class수 확인 -> 120가지의 개 품종 분류                                      
num_classes=metadata.features['label'].num_classes
print(num_classes)


# 입력이미지 크기를 224, [0,1]로 스케일 설정하고,  layer에 처리하기 
IMG_SIZE=224

resize_and_rescale=tf.keras.Sequential([
                                        layers.experimental.preprocessing.Resizing(IMG_SIZE, IMG_SIZE),
                                        layers.experimental.preprocessing.Rescaling(1./255)   # 입력 이미지를 [0,1]로 scaling
])

# 모델 컴파일 목적으로 batch 크기를 작게 설정
batch_size = 4

# classification을 위한 각 class one-hot 인코딩
train_ds = train_ds.map(lambda x, y: (resize_and_rescale(x), tf.one_hot(y, depth=120))).batch(batch_size)
val_ds = val_ds.map(lambda x, y: (resize_and_rescale(x), tf.one_hot(y, depth=120))).batch(batch_size)
test_ds = test_ds.map(lambda x, y: (resize_and_rescale(x), tf.one_hot(y, depth=120))).batch(batch_size)

# 입력 이미지 개수 None으로 처리된 것을 확인
# 출력 결과 class 수=120 
train_ds

# keras layers&models importing
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential

model = Sequential()

# 기존 모델에서 feature를 추출하는 layers의 학습 차단
for layer in InceptionV3.layers:
    layer.trainable= False
    
model.add(InceptionV3)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.2))
model.add(Dense(120,activation='softmax'))
model.summary()

# model compile -> loss함수 설정시 categorical_crossentropy / sarse_categorical_crossentropy / binary_crossentropy 구분
model.compile(optimizer= keras.optimizers.Adam(lr= 0.0001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

# train data size=12,000개 중 train, val size 명시
nb_train_samples = 9600.  
nb_valid_samples = 2400.

# model train
history = model.fit(
    train_ds, 
    epochs = 2,
    steps_per_epoch = nb_train_samples//batch_size,
    validation_data = val_ds, 
    validation_steps = nb_valid_samples//batch_size,
    verbose = 1,
    shuffle = True
)

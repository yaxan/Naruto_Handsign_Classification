import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceExtended
CollectiveAllReduceExtended._enable_check_health = False

tf.compat.v1.disable_eager_execution()
PATH_TO_TRAIN_DATA = "D:\\code\\Naruto_Handsign_Classification\\TrainDataa"
PATH_TO_TEST_DATA = "D:\\code\\Naruto_Handsign_Classification\\Data"
BATCH_SIZE = 16
TRAIN_SIZE = 3318
TEST_SIZE = 290
tf.random.set_seed(42)
np.random.seed(42)

#Descriptor of file structure of a dataset (example dataset_train)
#./dataset
# ->class 1
# ->class 2
#  .
#  .

if __name__ ==  '__main__':
  def get_datagen(dataset, aug=False):
      if aug:
          datagen = ImageDataGenerator(
                              rescale=1./255,
                              featurewise_center=False,
                              featurewise_std_normalization=False,
                              rotation_range=25,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.10,
                              zoom_range=0.15,
                              horizontal_flip=True,
                              fill_mode="nearest")
      else:
          datagen = ImageDataGenerator(rescale=1./255)

      return datagen.flow_from_directory(
              dataset,
              target_size=(224, 224),
              color_mode='rgb',
              shuffle = True,
              class_mode='categorical',
              batch_size=BATCH_SIZE)

  train_generator  = get_datagen(PATH_TO_TRAIN_DATA, False)
  test_generator   = get_datagen(PATH_TO_TEST_DATA, False)

  def load_model(m):
    if m == 'MN':
      model = tf.keras.applications.mobilenet_v2.MobileNetV2(
      input_shape=(224,224,3), alpha=1.0, include_top=False, weights='imagenet', pooling=None)

      for i in range(135):
       model.layers[i].trainable = False

      mobile_net = Flatten()(model.output)
      mobile_net = Dropout(0.5)(mobile_net)
      mobile_net = Dense(4096, activation='relu')(mobile_net)
      mobile_net = Dropout(0.5)(mobile_net)
      mobile_net = Dense(1024, activation='relu')(mobile_net)
      mobile_net = Dropout(0.5)(mobile_net)
      mobile_net = Dense(12, activation='softmax')(mobile_net)

      mobile_net_mobile = Model(model.input, mobile_net, name='Altered_MobileNet')
      mobile_net_mobile.summary()

      model = mobile_net_mobile

    else:
      model = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling=None,
        classes=12,
      )
      model.summary()
      # model = tf.keras.models.load_model("squeezenet")
      # model.summary()

      # for i in range(53):
      #   model.layers[i].trainable = False

      # squeezenet = Flatten()(model.layers[-2].output)
      # squeezenet = Dropout(0.5)(squeezenet)
      # squeezenet = Dense(4096, activation='relu')(squeezenet)
      # squeezenet = Dropout(0.5)(squeezenet)
      # squeezenet = Dense(1024, activation='relu')(squeezenet)
      # squeezenet = Dropout(0.5)(squeezenet)
      # squeezenet = Dense(12, activation='softmax')(squeezenet)

      # squeezenet_out = Model(model.input, squeezenet, name='Altered_SqueezeNet')
      # squeezenet_out.summary()

      # model = squeezenet_out
    return model

  model = load_model('MN')

  adam = tf.keras.optimizers.Adam(learning_rate=0.001)
  sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
  rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',mode='max',factor=0.5, patience=10, min_lr=0.001, verbose=1)
  early_stopper = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,
                                            mode='auto', baseline=None, restore_best_weights=False)
  model.compile(loss='categorical_crossentropy',
                  optimizer=adam, metrics=['accuracy'])

  history = model.fit(
      train_generator,
      validation_data=test_generator, 
      steps_per_epoch=TRAIN_SIZE// BATCH_SIZE,
      validation_steps=TEST_SIZE// BATCH_SIZE,
      shuffle=True,
      epochs=100,
      #callbacks=[early_stopper],
      use_multiprocessing=False,
  )

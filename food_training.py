'''
Program used to train transfer learning model that uses Inception V3 and retrains the last 173 layers of the model to fit the dataset. 
This one runs on my lab's workstation, which has an NVIDIA GTX 1080 TI for GPU, as opposed to my GTX 980. 
There is a substantial difference in time spent training between the GPUs.
'''

import os, sys
#import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime
import platform
import glob
import shutil
import tensorflow as tf


# from IPython.display import display
# from PIL import Image
from keras import backend as K
from keras import regularizers
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout, Activation, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib

from PIL import Image

#WIDTH = 299, HEIGHT = 299
NUM_EPOCHS = 1
BATCH_SIZE = 32
NUM_CLASSES = 451


FILEPATH = '../history_training/'
TRAINING_DIR = '../training_data/'
TESTING_DIR = '../testing_data/'

#found using: find DIR_NAME -type f | wc -l       --from stack overflow
TRAIN_SIZE = 166580
TEST_SIZE = 60990



##################################
def find_directory_number(directory):
    #return most recent directory number/experiment number by sorting and finding biggest one
    if len(os.listdir(directory)) == 0:
        return '0'
    else:
        dirs = os.listdir(directory)
        dirs_int = [int(x) for x in dirs]
        dirs_int.sort()
        print("sorted list: ", dirs_int)
        return str(dirs_int[-1])



#removes all training history files from directory. used for resetting training. 
def clean_directory(directory):
    source = directory
    for files in os.listdir(source):
        file_path = os.path.join(source, files)
		#try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
		#except Exception as e:
        #print(e)
#####################################################





# transfer learning to adapt it to dataset classes
# tried so far:
# --maxpooling2d with pool sizes of 4,4 and 8,8 with dropout of 0.5
# --avgpooling2d with pool sizes of 8,8 with dropout 0.4
# --avgpooling2d with pool sizes of 4,4 with no dropout layer
# --avgpooling2d with pool sizes of 2,2 with no dropout layer
# 
# -results show that average pooling2d is superior to maxpooling2d by a significant amount
# -between avgpooling2d and maxpooling2d, difference in validation accuracy is about .20
# -eliminating dropout layer when using avgpooling2d had little effect on accuracy
# ---this makes sense since dropout is used to prevent overfitting of large models but 
# ---we have a lot of data in this case, and the results show no overfitting at all

def last_layer_insertion(base_model, num_classes): #aka top layers of transfer learning model
    x = base_model.output
    x = AveragePooling2D(pool_size=(2,2))(x) #try different values of pool size and maxpooling
    # x = Dropout(0.5)(x) <----- experimentation shows that the dropout layer is unnecessary
    x = Flatten()(x)
    predictions = Dense(num_classes, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0005), activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def main(args):
    print("started program. python version:")
    print(platform.python_version())


    num_epochs = int(args.nb_epoch)
    batch = BATCH_SIZE
    num_classes = NUM_CLASSES

    training_dir = TRAINING_DIR
    testing_dir = TESTING_DIR
    num_training = TRAIN_SIZE
    num_testing = TEST_SIZE

    train_datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        horizontal_flip=True,
        vertical_flip=False,
        shear_range=0.1,
        rescale=1./255)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        training_dir,
        target_size=(299,299),
        batch_size=batch,
        class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(
        testing_dir,
        target_size=(299,299),
        batch_size=batch,
        shuffle=False,
	class_mode='categorical')



    base_model = InceptionV3(
    	weights='imagenet',
    	include_top=False, 
    	input_tensor=Input(shape=(299,299,3)))

    model = last_layer_insertion(base_model, num_classes)


    if args.clean_reset:
    	print('deleting previous training history...')
    	clean_directory(FILEPATH)


    #saving purposes. find most recent experiment and make new folder to save model in
    training_number = find_directory_number(FILEPATH)
    SAVEPATH = FILEPATH + str(int(training_number)+1)+'/'
    os.mkdir(SAVEPATH)
    print('training number: ' + training_number)


    #model compilation
    model.compile(
    	optimizer=SGD(lr=0.01, momentum=0.9), 
    	loss='categorical_crossentropy',
    	metrics=['acc'])


    #model checkpoint to save weights for each epoch
    filepath = SAVEPATH + "weights-{epoch:02d}-{val_acc:.2f}.hdf5"
    print("saving to filepath of " + filepath)
    checkpoint = ModelCheckpoint(
    	filepath,
    	monitor='val_acc',
    	verbose=1,
    	save_best_only=False,
    	save_weights_only=False,
    	mode='max')

    #saving logs for tensorboard data
    logdir = "../logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=logdir)


    #if using sgd optimizer, it's recommended to use a learning rate scheduler
    def schedule(epoch):
        if epoch < 15:
            return .01
        elif epoch < 25:
            return .002
        elif epoch < 35:
            return .0004
        elif epoch < 45:
            return .00008
        elif epoch < 55:
            return .000016
        else:
            return .0000032
    
    learning_rate_schedule = LearningRateScheduler(schedule)



    model.fit_generator(
    	train_generator,
    	validation_data=test_generator,
    	steps_per_epoch=(num_training // batch),
    	epochs=num_epochs,
    	validation_steps=(num_testing // batch),
    	callbacks=[tensorboard_callback, checkpoint, learning_rate_schedule],
    	verbose=1)


    #used for checking confusion matrix and shuffling data.
    #need to check if predict_generator can work without redoing training.
    pred= model.predict_generator(test_generator, num_testing // batch)
    predicted_class_indices=np.argmax(pred,axis=1)
    labels = (test_generator.class_indices)
    labels2 = dict((v,k) for k,v in labels.items())
    predictions = [labels2[k] for k in predicted_class_indices]
    print(confusion_matrix(predicted_class_indices, labels2))
    print("done")

'''
result: 95% accuracy and 86% validation accuracy. decent enough to start.
analysis and verification in validatingAndVisualization.py
'''


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--nb_epoch", "-e", default=NUM_EPOCHS)
    args.add_argument('--clean_reset','-c', action='store_true')
    args = args.parse_args()
    main(args)

#! /usr/local/bin/python

import tensorflow as tf
import numpy as np
import sys
import json
import math
import matplotlib.pylab as plt
import matplotlib
import augment

RADIUS_THRESHOLD=5.656      # this is 0.5 * sqrt ( 8^2 + 8^2 ) 

X_split_train=None 
Y_split_train=None
training_dataset_length=None
X_split_test=None 
Y_split_test=None
test_dataset_length=None


def voltamo_load_raw_training_dataset(path='./data'):
    global X_split_train, Y_split_train, training_dataset_length

    X_split_train = np.load(path+'/'+'X_split_train.npy')

    with open(path+'/'+'Y_split_train.npy', 'r') as f:
        Y_split_train = json.loads(f.read())

    training_dataset_length = len(X_split_train)

def voltamo_load_raw_test_dataset(path='./data'):
    global X_split_test, Y_split_test, test_dataset_length

    X_split_test = np.load(path+'/'+'X_split_test.npy')

    with open(path+'/'+'Y_split_test.npy', 'r') as f:
        Y_split_test = json.loads(f.read())

    test_dataset_length = len(X_split_test)

def voltamo_load_raw_datasets(path='./data', useAugmented = False):
    global X_split_train, Y_split_train, training_dataset_length
    global X_split_test, Y_split_test, test_dataset_length

    if (not useAugmented):
        voltamo_load_raw_training_dataset(path)
        voltamo_load_raw_test_dataset(path)
    else:
        X_split_train, Y_split_train, X_split_test, Y_split_test = augment.load_augmented_dataset()
        training_dataset_length = len(X_split_train)
        test_dataset_length = len(X_split_test)


def voltamo_build_segmentation_dataset(isTraining=True,imageSize=96):

    X_dataset = []
    Y_dataset = []
    all_boxes = []
    all_labels = []

    for k in range(training_dataset_length if isTraining else test_dataset_length):

        vlabel = np.zeros((12,12,9),dtype=np.float32)
        vlabel[:,:,0] = 1
        box_set = []
        label_set = []

        for bbox in (Y_split_train[k]['boundingBoxes'] if isTraining else Y_split_test[k]['boundingBoxes']) :
            label = bbox['label']
            x = bbox['x']
            y = bbox['y']
            w = bbox['w']
            h = bbox['h']
            r = math.sqrt( math.pow(w,2) + math.pow(h,2) ) * 0.5
            cx = x + w * 0.5
            cy = y + h * 0.5
            ix = int(cy) // 8
            iy = int(cx) // 8

            if (label == 1):
                #vlabel[ix,iy] = np.array([0,1,0])
                if (r <= RADIUS_THRESHOLD*1.5):
                    vlabel[ix,iy] = np.array([0,1,0,0,0,0,0,0,0])
                elif ( (r > RADIUS_THRESHOLD*1.5) and (r <= RADIUS_THRESHOLD*3.5) ):
                    vlabel[ix,iy] = np.array([0,0,1,0,0,0,0,0,0])
                elif ( (r > RADIUS_THRESHOLD*3.5) and (r <= RADIUS_THRESHOLD*5.5) ):
                    vlabel[ix,iy] = np.array([0,0,0,1,0,0,0,0,0])
                else: 
                    vlabel[ix,iy] = np.array([0,0,0,0,1,0,0,0,0])
            elif (label == 2): 
                #vlabel[ix,iy] = np.array([0,0,1])
                if (r <= RADIUS_THRESHOLD*1.5):
                    vlabel[ix,iy] = np.array([0,0,0,0,0,1,0,0,0])
                elif ( (r > RADIUS_THRESHOLD*1.5) and (r <= RADIUS_THRESHOLD*3.5) ):
                    vlabel[ix,iy] = np.array([0,0,0,0,0,0,1,0,0])
                elif ( (r > RADIUS_THRESHOLD*3.5) and (r <= RADIUS_THRESHOLD*5.5) ):
                    vlabel[ix,iy] = np.array([0,0,0,0,0,0,0,1,0])
                else:    
                    vlabel[ix,iy] = np.array([0,0,0,0,0,0,0,0,1])

            box_set.append([y/imageSize, x/imageSize, (y+h)/imageSize, (x+w)/imageSize])
            label_set.append(vlabel[ix,iy][1:])

        img = X_split_train[k] if isTraining else X_split_test[k]   

        Y_dataset.append(vlabel)        
        X_dataset.append(img)
        all_boxes.append(box_set)
        all_labels.append(label_set)


    if (isTraining):
        dataset = tf.data.Dataset.from_tensor_slices( ( tf.convert_to_tensor(X_dataset,np.float32), 
                                                        tf.convert_to_tensor(Y_dataset,np.float32) ) )
    else:
        dataset = tf.data.Dataset.from_tensor_slices( ( tf.convert_to_tensor(X_dataset,np.float32), 
                                                        tf.convert_to_tensor(Y_dataset,np.float32),
                                                        ( tf.ragged.constant(all_boxes), tf.ragged.constant(all_labels )) ) ) 

    dataset = dataset.batch(30)      
    return dataset

def voltamo_get_segmentation_datasets(path='./data', useAugmented = False):
    voltamo_load_raw_datasets(path, useAugmented)
    training = voltamo_build_segmentation_dataset(isTraining=True)
    validation = voltamo_build_segmentation_dataset(isTraining=False)

    return training, validation


def plot_image(img,title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    img = np.squeeze(img)
    ax.imshow(img, cmap=plt.cm.gray)
    # Draw Grid
    for i in range(12):
        x = [i*8, i*8]
        y = [0, 95]
        ax.plot(x, y, color="gray", linewidth=1)
    for j in range(12):
        x = [0, 95]
        y = [j*8, j*8]
        ax.plot(x, y, color="gray", linewidth=1)

    if (title):
        plt.text(0,94,f"{title}",size=12,color="darkblue",fontweight="bold")    

    x = [0,48]
    y = [0,48]      
    ax.plot(x, y, color="red", linewidth=1)

    plt.show()


def get_test_dataset(data_dir="./testing",size=(96,96)):
    return tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        labels=None,
        color_mode='grayscale',
        crop_to_aspect_ratio=True,
        image_size=size)

if (__name__ == "__main__"):

    training_dataset, validation_dataset = voltamo_get_segmentation_datasets()

    np.set_printoptions(threshold=sys.maxsize)

#    for element in training_dataset:
#        print(element)

#    for element in validation_dataset:
#        print(element)

    for a,b in training_dataset.take(1):
        print(a.shape)
        print(b.shape)

    for a,b,c in validation_dataset.take(1):
        print(a.shape)
        print(b.shape)
        print(f'( {c[0].shape}, {c[1].shape} )')

#    for a,b in training_dataset:        
#        for img in a:
#            plot_image(img)

    #for tds in get_test_dataset():
    #    for img in tds:
    #        plot_image(img,title=None)



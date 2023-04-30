#! /usr/local/bin/python

import sys
import json
import math
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa 
import cv2
import random
import os

LABELS = None
BB = None
NUM_SAMPLES = 0

X_train=None
Y_train=None
training_dataset_length=None
X_val=None
Y_val=None
validation_dataset_length=None
sample_id = None

def load_bounding_boxes(path = "./training/"):
    global LABELS, BB, NUM_SAMPLES

    with open(path+'bounding_boxes.labels', 'r') as f:
        LABELS = json.loads(f.read())
    BB = [] 
    for item in LABELS['boundingBoxes'].items():
        BB.append([item[0],item[1]])
    NUM_SAMPLES = len(BB)

def rotation_matrix(angle):
    a = (angle * math.pi / 180)
    return np.array([ [math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)] ])

def rotate_image(image, angle):
    a = (angle * math.pi / 180)
    return tfa.image.rotate(image, a, interpolation='bilinear', fill_mode='nearest').numpy()

def rotate_bounding_box(bbox,img,angle):
    shape = img.shape
    image_height = shape[0]
    image_width  = shape[1]

    x = bbox['x']
    y = bbox['y']
    w = bbox['width']
    h = bbox['height']

    image_center = np.array([ [image_width/2], [image_height/2] ])
    # corners = np.array ([[x,y], [x+w,y], [x+w,y+h],[x,y+h]]) - image_center
    corners = np.array ([ [x,x+w,x+w,x], [y,y,y+h,y+h]]) - image_center
    rotated_corners = np.dot(rotation_matrix(angle).T,corners) + image_center
    return rotated_corners.astype(int)

def optimized_bounding_box(R):
    xmin = min(R[0][0],R[0][1],R[0][2],R[0][3])
    ymin = min(R[1][0],R[1][1],R[1][2],R[1][3])
    xmax = max(R[0][0],R[0][1],R[0][2],R[0][3])
    ymax = max(R[1][0],R[1][1],R[1][2],R[1][3])
    return (xmin,ymin,xmax-xmin,ymax-ymin)


def draw_bounding_box(rot_img,R,OBB=None):
    color_green = (0,255,0)
    color_yellow = (0,255,255)    
    cv2.line(rot_img,(R[0][0],R[1][0]),(R[0][1],R[1][1]),color_green,4)
    cv2.line(rot_img,(R[0][1],R[1][1]),(R[0][2],R[1][2]),color_green,4)
    cv2.line(rot_img,(R[0][2],R[1][2]),(R[0][3],R[1][3]),color_green,4)
    cv2.line(rot_img,(R[0][3],R[1][3]),(R[0][0],R[1][0]),color_green,4)
    if (OBB != None):
        x,y,w,h = OBB
        cv2.line(rot_img,(x,y),(x+w,y),color_yellow,4)
        cv2.line(rot_img,(x+w,y),(x+w,y+w),color_yellow,4)
        cv2.line(rot_img,(x+w,y+w),(x,y+w),color_yellow,4)
        cv2.line(rot_img,(x,y+w),(x,y),color_yellow,4)

def shrink_image(img,target_size):

    img1 = tf.image.rgb_to_grayscale(img)

    img2 = tf.image.resize_with_pad(
                                    img1,
                                    target_height=target_size[1],
                                    target_width=target_size[0],
                                    method='nearest',
                                    antialias=True)

    return img2.numpy() / 256

def shrink_image_scale(img, target_size):

    original_height = img.shape[0]
    original_width = img.shape[1]
    original_channels = img.shape[2]

    pad_horizontally = True if (original_width < original_height) else False
    pad_size = int(abs(original_height - original_width) / 2)

    if pad_horizontally:
        colorLeft = np.mean(img[:,0,:]).astype(int)
        colorRight = np.mean(img[:,-1,:]).astype(int)
        pad = np.ones((original_height,pad_size,original_channels))
        draft_img = np.concatenate([pad*colorLeft,img,pad*colorRight],axis=1)
    else:
        colorTop = np.mean(img[0,:,:]).astype(int)
        colorBottom = np.mean(img[-1,:,:]).astype(int)
        pad = np.ones((pad_size,original_width,original_channels))
        draft_img = np.concatenate([pad*colorTop,img,pad*colorBottom],axis=0)

    scale = (target_size[1] / draft_img.shape[1])

    offset = (0,pad_size) if pad_horizontally else (pad_size,0) # offset = (vertical_offset, horizontal_offset)

    return ( shrink_image(draft_img,target_size), scale , offset)


def test(index=0, angle=0, path = "./training/"):

    np.set_printoptions(threshold=sys.maxsize)
    
    load_bounding_boxes(path)

    image_file=BB[index][0]
    bbox = BB[index][1]

    img = cv2.imread(path+image_file)
    rot_img = rotate_image(img,angle)

    for bbox in BB[index][1]:
        color = (0,255,255)        
        R = rotate_bounding_box(bbox,img,angle)
        OBB = optimized_bounding_box(R)
        draw_bounding_box(rot_img,R,OBB)


    cv2.imshow('image',rot_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # img = shrink_image(rot_img,(96,96))
    img, scale, offset = shrink_image_scale(rot_img, (96,96))
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(f'original shape={rot_img.shape}')
    print(f'shrinked shape={img.shape}')
    print(f'scale={scale}\toffset={offset}')    

def reset_augmented_dataset():
    global X_train , Y_train, training_dataset_length
    global X_val , Y_val, validation_dataset_length        
    global sample_id

    print(f'Reseting Dataset')
    X_train = []
    Y_train = []
    training_dataset_length = 0
    X_val = []
    Y_val = []
    validation_dataset_length = 0
    sample_id = 0


def build_augmented_dataset(path="./training/", target_size=(96,96), angle_list=None):

    global X_train , Y_train, training_dataset_length
    global sample_id

    load_bounding_boxes(path)

    for k in range(NUM_SAMPLES):
        image_file = BB[k][0]
        img = cv2.imread(path+image_file)
        print(f'file={image_file}\tshape={img.shape}')

        for angle in angle_list:
            rot_img = rotate_image(img,angle)
            shr_img, scale, offset = shrink_image_scale(rot_img, target_size)

            X_train.append(shr_img)

            sample = dict()
            sample['sampleId'] = sample_id
            sample_id += 1
            sample['boundingBoxes'] = []

            for bbox in BB[k][1]:
                R = rotate_bounding_box(bbox,img,angle)
                OBB = optimized_bounding_box(R)
                label = 1 if (bbox['label'] == 'O') else 2
                x = int((OBB[0]+offset[1])*scale)
                y = int((OBB[1]+offset[0])*scale)
                # adjusts width and height due to rotation
                # w = int(OBB[2]*scale) 
                # h = int(OBB[3]*scale)
                # keeps the same width and height
                w = int(bbox['width']*scale)
                h = int(bbox['height']*scale)
                sample['boundingBoxes'].append({ 'label' : label, 'x' : x, 'y' : y, 'w' : w, 'h' : h })

            Y_train.append(sample)
            training_dataset_length +=1
            print(f'\tangle={angle}\t{sample}')


def split_augmented_dataset(split_rate=0.1):
    global X_train , Y_train, training_dataset_length
    global X_val , Y_val, validation_dataset_length        

    print('Splitting Dataset (Training and Validation)')
    # Shuffle Dataset
    N = training_dataset_length
    seq = random.sample(range(N), N)

    # Create Training and Validation Sequences
    N_split = int(N * split_rate)
    val_seq = seq[:N_split]
    train_seq = seq[N_split:]

    # Rearange Datasets
    Shuffled_X_train = [ X_train[k] for k in train_seq ]
    Shuffled_Y_train = [ Y_train[k] for k in train_seq ]    
    Shuffled_X_val   = [ X_train[k] for k in val_seq ]
    Shuffled_Y_val   = [ Y_train[k] for k in val_seq ]    

    X_train = Shuffled_X_train
    Y_train = Shuffled_Y_train
    X_val   = Shuffled_X_val
    Y_val   = Shuffled_Y_val

def save_augmented_dataset(path = "./augmented"):

    if (not os.path.isdir(path)):
        os.mkdir(path)

    with open(path+'/X_train.npy','wb') as outfile:
        np.save(outfile, X_train)
    
    with open(path+'/Y_train.json','w') as outfile:
        json.dump(Y_train, outfile)

    with open(path+'/X_val.npy','wb') as outfile:
        np.save(outfile, X_val)

    with open(path+'/Y_val.json','w') as outfile:
        json.dump(Y_val, outfile)


def load_augmented_dataset(path = "./augmented"):

    X_train = np.load(path+'/X_train.npy')

    with open(path+'/Y_train.json','r') as f:
        Y_train = json.loads(f.read())

    X_val = np.load(path+'/X_val.npy')

    with open(path+'/Y_val.json','r') as f:
        Y_val = json.loads(f.read())

    return ( X_train, Y_train, X_val, Y_val )






if (__name__ == "__main__"):

#    test(index=47,angle=30)

    path = "./training/"
    load_bounding_boxes(path)

#    angle = 5.0

#    for k in range(NUM_SAMPLES):
#        image_file = BB[k][0]
#        img = cv2.imread(path+image_file)
#        print(f'file={image_file}\tshape={img.shape}')
#        for bbox in BB[k][1]:
#            print(f'\t {bbox}')            
#            R = rotate_bounding_box(bbox,img,angle)
#            OBB = optimized_bounding_box(R)            
#            print(f'\toptimized bbox={OBB}')

    reset_augmented_dataset()
    build_augmented_dataset(angle_list=[0,-10,10,-20,20,-30,30])
    split_augmented_dataset()    
    save_augmented_dataset()

#    X_train, Y_train, X_val, Y_val = load_augmented_datasets(path = "./augmented")

    print (f'{len(X_train)} Training Samples')
    print (f'{len(X_val)} Validation Samples')
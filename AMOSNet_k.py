#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 20 Oct 2020

@author: mihnea
"""

import caffe
import numpy as np
import csv
import cv2

total_Query_Images = 172 #number of query images in dataset
total_Ref_Images = 172 #number of reference images in dataset
ref_index_offset = 0
query_index_offset = 0

k = 5 #sequence length

# NOTE: Update the query and reference image paths below to point to your own dataset
query_directory = '/home/mihnea/datasets/nordland/live/'
ref_directory = '/home/mihnea/datasets/nordland/memory/'


def get_query_image_name(query):
    query_name = str(query + query_index_offset)

    return query_name + '.png'

def get_ref_image_name(ref):
    ref_name = str(ref + ref_index_offset)

    return ref_name + '.png'

#Create feature descriptors from the reference images
def compute_map_features(ref_map):   
    mean_npy = np.load('AMOSNet/amosnet_mean.npy') # Input numpy array
    print('Mean Array Shape:' + str(mean_npy.shape))
    net = caffe.Net('AMOSNet/deploy.prototxt', 'AMOSNet/AmosNet.caffemodel', caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    print(net.blobs['data'].data.shape)
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean_npy)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
    ref_features=[]
    features_ref_local=np.zeros((256,30))
    for image_reference in ref_map:
        image_reference = image_reference / 255.
        image_reference = image_reference[:,:,(2,1,0)]
        
        features_ref_local=np.zeros((256,30))

        if(image_reference is not None):
            
            transformed_image_ref = transformer.preprocess('data', image_reference)
            net.blobs['data'].data[...] = transformed_image_ref.copy()
            out = net.forward()
            features_ref=np.asarray(net.blobs['conv5'].data)[1,:,:,:].copy()

            for i in range(256):

                #S=1
                features_ref_local[i,0]=np.max(features_ref[i,:,:])
                
                #S=2
                features_ref_local[i,1]=np.max(features_ref[i,0:6,0:6])            
                features_ref_local[i,2]=np.max(features_ref[i,0:6,7:12])
                features_ref_local[i,3]=np.max(features_ref[i,7:12,0:6])
                features_ref_local[i,4]=np.max(features_ref[i,7:12,7:12])
                
                #S=3
                features_ref_local[i,5]=np.max(features_ref[i,0:4,0:4])
                features_ref_local[i,6]=np.max(features_ref[i,0:4,5:8])
                features_ref_local[i,7]=np.max(features_ref[i,0:4,9:12])
                features_ref_local[i,8]=np.max(features_ref[i,5:8,0:4])
                features_ref_local[i,9]=np.max(features_ref[i,5:8,5:8])
                features_ref_local[i,10]=np.max(features_ref[i,5:8,9:12])
                features_ref_local[i,11]=np.max(features_ref[i,9:12,0:4])
                features_ref_local[i,12]=np.max(features_ref[i,9:12,5:8])
                features_ref_local[i,13]=np.max(features_ref[i,9:12,9:12])
    
                #S=4
                features_ref_local[i,14]=np.max(features_ref[i,0:3,0:3])
                features_ref_local[i,15]=np.max(features_ref[i,0:3,4:6])
                features_ref_local[i,16]=np.max(features_ref[i,0:3,7:9])
                features_ref_local[i,17]=np.max(features_ref[i,0:3,10:12])
                features_ref_local[i,18]=np.max(features_ref[i,4:6,0:3])
                features_ref_local[i,19]=np.max(features_ref[i,4:6,4:6])
                features_ref_local[i,20]=np.max(features_ref[i,4:6,7:9])
                features_ref_local[i,21]=np.max(features_ref[i,4:6,10:12])
                features_ref_local[i,22]=np.max(features_ref[i,7:9,0:3])
                features_ref_local[i,23]=np.max(features_ref[i,7:9,4:6])
                features_ref_local[i,24]=np.max(features_ref[i,7:9,7:9])
                features_ref_local[i,25]=np.max(features_ref[i,7:9,10:12])
                features_ref_local[i,26]=np.max(features_ref[i,10:12,0:3])
                features_ref_local[i,27]=np.max(features_ref[i,10:12,4:6])
                features_ref_local[i,28]=np.max(features_ref[i,10:12,7:9])
                features_ref_local[i,29]=np.max(features_ref[i,10:12,10:12])

            ref_features.append(features_ref_local)
    print('Reference images descriptors computed!')

    return ref_features

#Create feature descriptors from the query images
def compute_query_desc(image_query):
    
    mean_npy = np.load('AMOSNet/amosnet_mean.npy') # Input numpy array
    print('Mean Array Shape:' + str(mean_npy.shape))
    net = caffe.Net('AMOSNet/deploy.prototxt', 'AMOSNet/AmosNet.caffemodel', caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    print(net.blobs['data'].data.shape)
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mean_npy)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR    
    features_query_local=np.zeros((256,30))

    image_query = image_query / 255.
    image_query = image_query[:,:,(2,1,0)]

    if (image_query is not None):
        
        transformed_image_query = transformer.preprocess('data', image_query)
        net.blobs['data'].data[...] = transformed_image_query.copy()
        out = net.forward()
        features_query=np.asarray(net.blobs['conv5'].data)[1,:,:,:].copy()
    
        features_query_local=np.zeros((256,30))
        
        for i in range(256):
        
                #S=1
                features_query_local[i,0]=np.max(features_query[i,:,:])
                
                #S=2
                features_query_local[i,1]=np.max(features_query[i,0:6,0:6])
                features_query_local[i,2]=np.max(features_query[i,0:6,7:12])
                features_query_local[i,3]=np.max(features_query[i,7:12,0:6])
                features_query_local[i,4]=np.max(features_query[i,7:12,7:12])
                
                #S=3
                features_query_local[i,5]=np.max(features_query[i,0:4,0:4])
                features_query_local[i,6]=np.max(features_query[i,0:4,5:8])
                features_query_local[i,7]=np.max(features_query[i,0:4,9:12])
                features_query_local[i,8]=np.max(features_query[i,5:8,0:4])
                features_query_local[i,9]=np.max(features_query[i,5:8,5:8])
                features_query_local[i,10]=np.max(features_query[i,5:8,9:12])
                features_query_local[i,11]=np.max(features_query[i,9:12,0:4])
                features_query_local[i,12]=np.max(features_query[i,9:12,5:8])
                features_query_local[i,13]=np.max(features_query[i,9:12,9:12])
    
                #S=4
                features_query_local[i,14]=np.max(features_query[i,0:3,0:3])
                features_query_local[i,15]=np.max(features_query[i,0:3,4:6])
                features_query_local[i,16]=np.max(features_query[i,0:3,7:9])
                features_query_local[i,17]=np.max(features_query[i,0:3,10:12])
                features_query_local[i,18]=np.max(features_query[i,4:6,0:3])
                features_query_local[i,19]=np.max(features_query[i,4:6,4:6])
                features_query_local[i,20]=np.max(features_query[i,4:6,7:9])
                features_query_local[i,21]=np.max(features_query[i,4:6,10:12])
                features_query_local[i,22]=np.max(features_query[i,7:9,0:3])
                features_query_local[i,23]=np.max(features_query[i,7:9,4:6])
                features_query_local[i,24]=np.max(features_query[i,7:9,7:9])
                features_query_local[i,25]=np.max(features_query[i,7:9,10:12])
                features_query_local[i,26]=np.max(features_query[i,10:12,0:3])
                features_query_local[i,27]=np.max(features_query[i,10:12,4:6])
                features_query_local[i,28]=np.max(features_query[i,10:12,7:9])
                features_query_local[i,29]=np.max(features_query[i,10:12,10:12])

    return features_query_local

# perform_VPR creates fixed-length (k) query/reference sequences and performs image sequence matching
def perform_VPR(features_query_local,ref_map_features, initial_query_image, initial_ref_image):

    sequential_score = 0  
    i = initial_query_image
    j = initial_ref_image

    while(i < initial_query_image + k and j < initial_ref_image + k):
        match_score = 1 - (np.sum(abs(np.subtract(features_query_local[i],ref_map_features[j])))/(256*256))
        sequential_score = sequential_score + match_score 
        i = i + 1
        j = j + 1
    sequential_score = sequential_score / k        

    return sequential_score


ref_list = []
feature_descriptors = []
for ref in range(total_Ref_Images):

    try:
        img_1 = cv2.imread(ref_directory + get_ref_image_name(ref))

    except (IOError, ValueError) as e:
        img_1 = None
        print('Exception! \n \n \n \n', ref)

    if (img_1 is not None):
        ref_list.append(img_1)


query_list = []
for query in range(total_Query_Images):

    try:
        img_2 = cv2.imread(query_directory + get_query_image_name(query))

    except (IOError, ValueError) as e:
        img_2 = None
        print('Exception! \n \n \n \n')

    if (img_2 is not None):
        query_list.append(img_2)

feature_descriptors = compute_map_features(ref_list)

query_descriptor = []
for i in range(len(query_list)):
    query_descriptor.append(compute_query_desc(query_list[i]))

for i in range(total_Query_Images - k + 1):
    ref_template_score = []
    for j in range(total_Ref_Images - k + 1):
         score = perform_VPR(query_descriptor, feature_descriptors, i, j)
         ref_template_score.append(score)

    #writing the results in a csv file
    with open('AMOSNet_Results_k_5.csv', 'a') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        row = str(i) + ',' + str(np.argmax(ref_template_score)) + ',' + str(np.amax(ref_template_score))
        my_writer.writerow([row])
    csvfile.close()



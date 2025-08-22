#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 20 Oct 2020

@author: mihnea
"""
import cv2
import numpy as np
import csv

total_Query_Images = 172 #number of query images in dataset
total_Ref_Images = 172 #number of reference images in dataset
ref_index_offset=0
query_index_offset=0

k = 5 #sequence length

# NOTE: Update the query and reference image paths below to point to your own dataset
query_directory='/home/mihnea/datasets/nordland/live/'
ref_directory='/home/mihnea/datasets/nordland/memory/'

def get_query_image_name(query):
    query_name = str(query + query_index_offset)
     
    return query_name + '.png'

def get_ref_image_name(ref):
    ref_name = str(ref + ref_index_offset)
    
    return ref_name + '.png'

#Create feature descriptors from the reference images
def compute_map_features(ref_map):
    
    winSize = (512,512)
    blockSize = (32,32)
    blockStride = (16,16)
    cellSize = (16,16)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    ref_desc_list=[]
    
    for ref_image in ref_map:
        
        if ref_image is not None:    
            hog_desc=hog.compute(cv2.resize(ref_image, winSize))
            
        ref_desc_list.append(hog_desc)
        
    return ref_desc_list

#Create feature descriptors from the query images
def compute_query_desc(query):
        
    winSize = (512,512)
    blockSize = (32,32)
    blockStride = (16,16)
    cellSize = (16,16)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    query_desc=hog.compute(cv2.resize(query, winSize))
    
    return query_desc

# perform_VPR creates fixed-length (k) query/reference sequences and performs image sequence matching
def perform_VPR(query_desc,ref_map_features, initial_query_image, initial_ref_image):
    sequential_score = 0
    i = initial_query_image
    j = initial_ref_image

    while(i < initial_query_image + k and j < initial_ref_image + k):
        score = np.dot(query_desc[i].T,ref_map_features[j])/(np.linalg.norm(query_desc[i])*np.linalg.norm(ref_map_features[j]))
        sequential_score = sequential_score + score
        i = i + 1
        j = j + 1

    sequential_score = sequential_score / k

    return sequential_score


ref_list = []
feature_descriptors = []

for ref in range(total_Ref_Images):
    try:
        img_1 = cv2.imread(ref_directory+get_ref_image_name(ref), 0)
    
    except (IOError, ValueError) as e:
        img_1 = None
        print('Exception! \n \n \n \n',ref)        
        
    if (img_1 is not None):
	    ref_list.append(img_1)


query_list = []

for query in range(total_Query_Images):
    try:    
        img_2 = cv2.imread(query_directory+get_query_image_name(query), 0)

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

    # writing the results in a csv file
    with open('Results_HOG_k_5.csv', 'a') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        row = str(i) + ',' + str(np.argmax(ref_template_score)) + ',' + str(np.amax(ref_template_score))
        my_writer.writerow([row])
    csvfile.close()

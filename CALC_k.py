"""
Created on 20 Oct 2020

@author: mihnea
"""

import caffe
import numpy as np
import cv2
import csv
import time

total_Query_Images = 172 #number of query images in dataset
total_Ref_Images = 172 #number of reference images in dataset
ref_index_offset = 0
query_index_offset = 0

k = 2 #sequence length

# NOTE: Update the query and reference image paths below to point to your own dataset
query_directory = '/home/mihnea/datasets/nordland/live/'
ref_directory = '/home/mihnea/datasets/nordland/memory/'


def get_query_image_name(query):
    query_name = str(query + query_index_offset)

    return query_name + '.png'


def get_ref_image_name(ref):
    ref_name = str(ref + ref_index_offset)

    return ref_name + '.png'


def computeForwardPasses(net, im, transformer, resize_net):
    """
    Compute the forward passes for CALC
    """

    img_yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    if im.shape[2] > 1:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if not resize_net:
        im = cv2.resize(im, (160, 120), interpolation = cv2.INTER_CUBIC)
    else:
        transformer = caffe.io.Transformer({'X1':(1,1,im.shape[0],im.shape[1])})    
        transformer.set_raw_scale('X1',1./255)
        
        x1 = net.blobs['X1']
        x1.reshape(1,1,im.shape[0],im.shape[1])
        net.reshape()

    net.blobs['X1'].data[...] = transformer.preprocess('X1', im)
    net.forward()
    d = np.copy(net.blobs['descriptor'].data[...])
    d /= np.linalg.norm(d)

    return d

#Create feature descriptors from the reference images
def compute_map_features(ref_map_images):
    
    net_def_path='CALC/proto/deploy.prototxt'
    net_model_path='CALC/model/calc.caffemodel'
    resize_net=False 
    caffe.set_mode_cpu()
               
    net=caffe.Net(net_def_path,1,weights=net_model_path)

    database = [] # stored pic descriptors   
    
    # Use caffe's transformer
    transformer = caffe.io.Transformer({'X1':(1,1,120,160)})    
    transformer.set_raw_scale('X1',1./255)

    for img in ref_map_images:    
        
        descr= computeForwardPasses(net, img, transformer, resize_net)        
        database.append(descr)         
    
    return database

#Create feature descriptors from the query images
def compute_query_desc(im):
    
    net_def_path='CALC/proto/deploy.prototxt'
    net_model_path='CALC/model/calc.caffemodel'
    resize_net=False 
    caffe.set_mode_cpu()

    net=caffe.Net(net_def_path,1,weights=net_model_path)

    # Use caffe's transformer
    transformer = caffe.io.Transformer({'X1':(1,1,120,160)})    
    transformer.set_raw_scale('X1',1./255)
    
    descr= computeForwardPasses(net, im, transformer, resize_net) 
    
    return descr

# perform_VPR creates fixed-length (k) query/reference sequences and performs image sequence matching
def perform_VPR(descr,ref_map_features, initial_query_image, initial_ref_image):
    sequential_score = 0
    i = initial_query_image
    j = initial_ref_image

    while(i < initial_query_image + k and j < initial_ref_image + k):
        curr_sim = np.dot(descr[i], ref_map_features[j].T)
        sequential_score = sequential_score + curr_sim
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
        print(('Exception! \n \n \n \n', ref))

    if img_1 is not None:
        ref_list.append(img_1)
        startencodetimer = time.time()


query_list = []
for query in range(total_Query_Images):
    try:
        img_2 = cv2.imread(query_directory + get_query_image_name(query))

    except (IOError, ValueError) as e:
        img_2 = None
        print('Exception! \n \n \n \n')

    if img_2 is not None:
        query_list.append(img_2)
        encodetime = time.time() - startencodetimer


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
    with open('CALC_Results_k_2.csv', 'a') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        row = str(i) + ',' + str(np.argmax(ref_template_score)) + ',' + str(np.amax(ref_template_score))
        my_writer.writerow([row])
    csvfile.close()

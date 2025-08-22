"""
Created on 20 Oct 2020

@author: mihnea
"""

import cv2
import numpy as np
import tensorflow as tf
import netvlad_tf.net_from_mat as nfm
import netvlad_tf.nets as nets
import time
import csv

total_Query_Images = 172 #number of query images in dataset
total_Ref_Images = 172 #number of reference images in dataset
ref_index_offset = 0
query_index_offset = 0

k = 3 #sequence length

# NOTE: Update the query and reference image paths below to point to your own dataset
query_directory = '/home/mihnea/datasets/nordland/live/'
ref_directory = '/home/mihnea/datasets/nordland/memory/'


def get_query_image_name(j):
    query_name = str(j + query_index_offset)
     
    return query_name + '.png'

def get_ref_image_name(j):
    ref_name = str(j + ref_index_offset)
    
    return ref_name + '.png'

#Create feature descriptors from the reference images
def compute_map_features(ref_map_images):
    ref_desc=[]
    tf.reset_default_graph()
    image_batch = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, 3])
    
    net_out = nets.vgg16NetvladPca(image_batch)
    saver = tf.train.Saver()
    
    sess = tf.Session()
    saver.restore(sess, nets.defaultCheckpoint())
    
    for img in ref_map_images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        batch = np.expand_dims(img, axis=0)
        t1 = time.time()
        desc = sess.run(net_out, feed_dict={image_batch: batch})
        print('Encode Time: ', time.time()-t1)
        ref_desc.append(desc)
        
    return ref_desc

#Create feature descriptors from the query images
def compute_query_desc(image_query):
    tf.reset_default_graph()
    image_batch = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, 3])
    
    net_out = nets.vgg16NetvladPca(image_batch)
    saver = tf.train.Saver()
    
    sess = tf.Session()
    saver.restore(sess, nets.defaultCheckpoint())
    
    batch = np.expand_dims(image_query, axis=0)
    query_desc = sess.run(net_out, feed_dict={image_batch: batch}) 
    
    return query_desc

# perform_VPR creates fixed-length (k) query/reference sequences and performs image sequence matching
def perform_VPR(query_desc, ref_map_features, initial_query_image, initial_ref_image):
    sequential_score = 0
    i = initial_query_image
    j = initial_ref_image

    while(i < initial_query_image + k and j < initial_ref_image + k):
        match_score = np.dot(query_desc[i], ref_map_features[j].T)
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

    # writing the results in a csv file
    with open('Results_NetVLAD_k_3.csv', 'a') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        row= str(i) + ',' + str(np.argmax(ref_template_score)) + ',' + str(np.amax(ref_template_score))
        my_writer.writerow([row])
    csvfile.close()
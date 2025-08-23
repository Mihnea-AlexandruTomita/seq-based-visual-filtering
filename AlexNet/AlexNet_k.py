import caffe
import numpy as np
import cv2
import csv
import os
from os.path import dirname

total_Query_Images = 100  # number of query images in dataset
total_Ref_Images = 100  # number of reference images in dataset
ref_index_offset = 0
query_index_offset = 0

k = 3  # sequence length

# NOTE: Update the query and reference image paths below to point to your own dataset
query_directory = '/home/mihnea/CoHOG_Results_RAL2019/campus_loop_original/live/'
ref_directory = '/home/mihnea/CoHOG_Results_RAL2019/campus_loop_original/memory/'


def get_query_image_name(query):
    query_name = str(query + query_index_offset)

    return query_name + '.jpg'


def get_ref_image_name(ref):
    ref_name = str(ref + ref_index_offset)

    return ref_name + '.jpg'


first_it = True
A = None

#Create feature descriptors from the reference images
def compute_map_features(ref_map_images):
    ref_map_images_features=[]
    
    alexnet_proto_path=os.path.join(dirname(__file__),"alexnet/deploy.prototxt")
    alexnet_weights=os.path.join(dirname(__file__),"alexnet/alexnet.caffemodel")
    alexnet = caffe.Net(alexnet_proto_path,1,weights=alexnet_weights)

    transformer_alex = caffe.io.Transformer({'data':(1,3,227,227)})    
    transformer_alex.set_raw_scale('data',1./255)
    transformer_alex.set_transpose('data', (2,0,1))
    transformer_alex.set_channel_swap('data', (2,1,0))
    
    for ref_img in ref_map_images:
            img_yuv = cv2.cvtColor(ref_img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            alex_conv3 = None      
    
            im2 = cv2.resize(im, (227,227), interpolation=cv2.INTER_CUBIC)
            alexnet.blobs['data'].data[...] = transformer_alex.preprocess('data', im2)
            alexnet.forward()
            alex_conv3 = np.copy(alexnet.blobs['conv3'].data[...])
            alex_conv3 = np.reshape(alex_conv3, (alex_conv3.size, 1))
            global first_it
            global A
            if first_it:
                np.random.seed(0)
                A = np.random.randn(1064, alex_conv3.size) # For Gaussian random projection  descr[0].size=1064
                first_it = False
            alex_conv3 = np.matmul(A, alex_conv3)
            alex_conv3 = np.reshape(alex_conv3, (1, alex_conv3.size))
            alex_conv3 /= np.linalg.norm(alex_conv3)

            ref_map_images_features.append(alex_conv3)
    
    return ref_map_images_features


#Create feature descriptors from the query images
def compute_query_desc(image_query):

    alexnet_proto_path=os.path.join(dirname(__file__),"alexnet/deploy.prototxt")
    alexnet_weights=os.path.join(dirname(__file__),"alexnet/alexnet.caffemodel")
    alexnet = caffe.Net(alexnet_proto_path,1,weights=alexnet_weights)

    transformer_alex = caffe.io.Transformer({'data':(1,3,227,227)})    
    transformer_alex.set_raw_scale('data',1./255)
    transformer_alex.set_transpose('data', (2,0,1))
    transformer_alex.set_channel_swap('data', (2,1,0))

    img_yuv = cv2.cvtColor(image_query, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    im = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    alex_conv3 = None      

    im2 = cv2.resize(im, (227,227), interpolation=cv2.INTER_CUBIC)
    alexnet.blobs['data'].data[...] = transformer_alex.preprocess('data', im2)
    alexnet.forward()
    alex_conv3 = np.copy(alexnet.blobs['conv3'].data[...])
    alex_conv3 = np.reshape(alex_conv3, (alex_conv3.size, 1))
    global first_it
    global A
    if first_it:
        np.random.seed(0)
        A = np.random.randn(1064, alex_conv3.size) # For Gaussian random projection  descr[0].size=1064
        first_it = False
    alex_conv3 = np.matmul(A, alex_conv3)
    alex_conv3 = np.reshape(alex_conv3, (1, alex_conv3.size))
    alex_conv3 /= np.linalg.norm(alex_conv3)
    
    return alex_conv3


# perform_VPR creates fixed-length (k) query/reference sequences and performs image sequence matching
def perform_VPR(alex_conv3, ref_map_features, initial_query_image, initial_ref_image):
    sequential_score = 0
    i = initial_query_image
    j = initial_ref_image

    while(i < initial_query_image + k and j < initial_ref_image + k):
        match_score = np.dot(alex_conv3[i], ref_map_features[j].T)
        sequential_score += match_score
        i = i + 1
        j = j + 1

    sequential_score /= k

    return sequential_score


feature_descriptors = []
ref_list = []
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
    with open('Results_AlexNet_k_3.csv', 'a') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        row = str(i) + ',' + str(np.argmax(ref_template_score)) + ',' + str(np.amax(ref_template_score))
        my_writer.writerow([row])
    csvfile.close()








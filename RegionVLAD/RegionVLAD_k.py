# Implementation of Region-VLAD VPR framework
# Ahmad Khaliq
# ahmedest61@hotmail.com
# Modified by Mubariz Zaffar
# Sequence-based filtering functionality added by Mihnea-Alexandru Tomita

from _collections import defaultdict
import caffe
import pickle
import numpy as np
from skimage.measure import regionprops,label
import itertools
import time
import os
import csv
import cv2
from os.path import dirname

startruntimer = time.time()

total_Query_Images = 172 #number of query images in dataset
total_Ref_Images = 172 #number of reference images in dataset
ref_index_offset = 0
query_index_offset = 0

k = 4 #sequence length

# NOTE: Update the query and reference image paths below to point to your own dataset
query_directory = '/home/mihnea/datasets/nordland/live/'
ref_directory = '/home/mihnea/datasets/nordland/memory/'

def get_query_image_name(query):
    query_name = str(query + query_index_offset)
     
    return query_name + '.png'

def get_ref_image_name(ref):
    ref_name = str(ref + ref_index_offset)
    
    return ref_name + '.png'


# Paths to protext, model and mean file
protxt = os.path.join(dirname(__file__),"AlexnetPlaces365/deploy_alexnet_places365.prototxt")
model = os.path.join(dirname(__file__),"AlexnetPlaces365/alexnet_places365.caffemodel")
mean = os.path.join(dirname(__file__),"AlexnetPlaces365/places365CNN_mean.binaryproto")

N = 400 #No. of ROIs
layer = 'conv3'
Features,StackedFeatures = defaultdict(list),defaultdict(list)
set_gpu =False
gpu_id = 0
totalT = 0

if set_gpu:
    caffe.set_mode_gpu()
    caffe.set_device(gpu_id)
else:
    caffe.set_mode_cpu()
    
    
# Load pickle file
def load_obj(name ):
    with open( name, 'rb') as f:
        return pickle.load(f)

# Read mean file
def binaryProto2npy(binaryProtoMeanFile):

     blob = caffe.proto.caffe_pb2.BlobProto()
     data = open( binaryProtoMeanFile, 'rb' ).read()
     blob.ParseFromString(data)
     data = np.array(blob.data)
     arr = np.array( caffe.io.blobproto_to_array(blob) )
     return arr[0]   

# Extract N ROIs from the conv layer
def getROIs(imgConvFeat,imgLocalConvFeat,img):
    
    clustersEnergies_Ej = []
    clustersBoxes  = []
    allROI_Box = [] 
    aggregatedROIs = []
    
    for featuremap in imgConvFeat:                         
        clusters = regionprops(label(featuremap),intensity_image=featuremap,cache=False)
        clustersBoxes.append(list(cluster.bbox for cluster in clusters))
        clustersEnergies_Ej.append(list(cluster.mean_intensity for cluster in clusters))

    # Make a list of ROIs with their bounded boxes
    clustersBoxes = list(itertools.chain.from_iterable(clustersBoxes))
    clustersEnergies_Ej = list(itertools.chain.from_iterable(clustersEnergies_Ej))

    # Sort the ROIs based on energies
    allROIs = sorted(clustersEnergies_Ej,reverse=True)

    # Pick up top N energetic ROIs with their bounding boxes
    allROIs = allROIs[:N]
    allROI_Box = [clustersBoxes[clustersEnergies_Ej.index(i)] for i in allROIs]
#    clustersEnergies_Ej.clear()
#    clustersBoxes.clear()
    aggregatedNROIs = np.zeros((N,imgLocalConvFeat.shape[2]))

    # Retreive the aggregated local descriptors lying under N ROIs
    for ROI in range(len(allROI_Box)):
  #      minRow, minCol, maxRow, maxCol = allROI_Box[ROI][0],allROI_Box[ROI][1],allROI_Box[ROI][2],allROI_Box[ROI][3]
        aggregatedNROIs[ROI,:]=np.sum(imgLocalConvFeat[allROI_Box[ROI][0]:allROI_Box[ROI][2],allROI_Box[ROI][1]:allROI_Box[ROI][3]],axis=(0,1))   #Sometimes elements come out as nan for rare images, which breaks the code. I have picked this up with the author and for the time being patched this. Maybe some issue with package versions. Needs further debugging. #Mubariz

    # NxK dimensional ROIs
    return np.asarray(aggregatedNROIs)

# Retreive the VLAD representation
def getVLAD(X,visualDictionary):
    
    predictedLabels = visualDictionary.predict(X)
    centers = visualDictionary.cluster_centers_
    labels=visualDictionary.labels_
    k=visualDictionary.n_clusters
   
    m,d = X.shape
    Vlad=np.zeros([k,d])
    #computing the differences
    # for all the clusters (visual words)
    for i in range(k):
        # if there is at least one descriptor in that cluster
        if np.sum(predictedLabels==i)>0:
            Vlad[i]=np.sum(X[predictedLabels==i,:]-centers[i],axis=0)  
    Vlad = Vlad.flatten()
    Vlad = np.sign(Vlad)*np.sqrt(np.abs(Vlad))
    Vlad = Vlad/np.sqrt(np.dot(Vlad,Vlad))        
    Vlad = Vlad.reshape(k,d)    
    return Vlad 

#Create feature descriptors from the reference images
def compute_map_features(ref_map_images):
        
    net = caffe.Net(protxt, model, caffe.TEST)
    batch_size = 1
    inputSize = net.blobs['data'].shape[2]
    net.blobs['data'].reshape(batch_size,3,inputSize,inputSize)
        
    ref_map_images_descs=[]
    
    # Configuration 1
    if N==200:
        V = 128
        vocab = load_obj(os.path.join(dirname(__file__),"Vocabulary/Vocabulary_100_200_300_Protocol2.pkl"))
    # Configuration 2
    else:
        V = 256
        vocab = load_obj(os.path.join(dirname(__file__),"Vocabulary/Vocabulary_400_Protocol2.pkl"))
        
    # Set Caffe
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    mean_file = binaryProto2npy(mean)
    transformer.set_mean('data', mean_file.mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255)
    itr=0
    for img in ref_map_images:

        net.blobs['data'].data[...] = transformer.preprocess('data', img/255.0)
  
        # Forward Pass
        res = net.forward()
    
        # Stack the activations of feature maps to make local descriptors
        Features[layer] = np.array(net.blobs[layer].data[0].copy())
        
        print(Features[layer].shape)
        StackedFeatures[layer]=Features[layer].transpose(1,2,0)
        print(StackedFeatures[layer].shape)
        # Retrieve N ROIs for test and ref images
        ROIs= getROIs(Features[layer],StackedFeatures[layer],img)
#        print(ROIs)
        
        vocabulary = vocab[N][V][layer]
        
        # Retrieve VLAD descriptors using ROIs and vocabulary
        VLAD= getVLAD(ROIs,vocabulary)
        
        ref_map_images_descs.append(VLAD)
        print(itr)
        itr=itr+1
        
    return ref_map_images_descs 
        
#Create feature descriptors from the query images
def compute_query_desc(image_query):
    image_query=image_query/255.0
    net = caffe.Net(protxt, model, caffe.TEST)
    batch_size = 1
    inputSize = net.blobs['data'].shape[2]
    net.blobs['data'].reshape(batch_size,3,inputSize,inputSize)

    # Configuration 1
    if N==200:
        V = 128
        vocab = load_obj(os.path.join(dirname(__file__),"Vocabulary/Vocabulary_100_200_300_Protocol2.pkl"))
    # Configuration 2
    else:
        V = 256
        vocab = load_obj(os.path.join(dirname(__file__),"Vocabulary/Vocabulary_400_Protocol2.pkl"))
        
    # Set Caffe
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    mean_file = binaryProto2npy(mean)
    transformer.set_mean('data', mean_file.mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255)
           
    net.blobs['data'].data[...] = transformer.preprocess('data', image_query)
        
    # Forward Pass
    res = net.forward()

    # Stack the activations of feature maps to make local descriptors
    Features[layer] = np.array(net.blobs[layer].data[0].copy())
    StackedFeatures[layer]=Features[layer].transpose(1,2,0)

    # Retrieve N ROIs for test and ref images
    ROIs= getROIs(Features[layer],StackedFeatures[layer],image_query)

    vocabulary = vocab[N][V][layer]
    
    # Retrieve VLAD descriptors using ROIs and vocabulary
    VLAD= getVLAD(ROIs,vocabulary)    
    
    return VLAD 

# perform_VPR creates fixed-length (k) query/reference sequences and performs image sequence matching
def perform_VPR(query_map_features, ref_map_features, initial_query_image, initial_ref_image):
    sequential_score = 0
    i = initial_query_image
    j = initial_ref_image

    while(i < initial_query_image + k and j < initial_ref_image + k):
            match_score = np.sum(np.einsum('ij,ij->i', query_map_features[i], ref_map_features[j]))
            sequential_score = sequential_score + match_score
            i = i + 1
            j = j + 1

    sequential_score = sequential_score / k	

    return sequential_score


ref_list = []
feature_descriptors = []
for ref in range(total_Ref_Images):
    try:
        img_1 = cv2.imread(ref_directory+get_ref_image_name(ref))
    
    except (IOError, ValueError) as e:
        img_1 = None
        print('Exception! \n \n \n \n',ref)        
        
    if (img_1 is not None):
        
	    ref_list.append(img_1)


feature_descriptors = compute_map_features(ref_list)

query_list = []
for query in range(total_Query_Images):
    
    try:    
        img_2 = cv2.imread(query_directory+get_query_image_name(query))

    except (IOError, ValueError) as e:
        img_2 = None
        print('Exception! \n \n \n \n')    
       
    if (img_2 is not None):
	    query_list.append(img_2)
   

feature_descriptors = compute_map_features(ref_list)

query_descriptor = []
for i in range(len(query_list)):
    query_descriptor.append(compute_query_desc(query_list[i]))


startmatchtimer = time.time()
for i in range(total_Query_Images - k + 1):
    ref_template_score = []
    for j in range(total_Ref_Images - k + 1):
        score = perform_VPR(query_descriptor, feature_descriptors, i , j)
        ref_template_score.append(score)
    matchingtime = time.time() - startmatchtimer
    print("matching time: " + str(matchingtime))

    # writing the results in a csv file
    with open('Results_RegionVLAD_k_4.csv', 'a') as csvfile:
        my_writer = csv.writer(csvfile, delimiter=',', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        row = str(i) + ',' + str(np.argmax(ref_template_score)) + ',' + str(np.amax(ref_template_score))
        my_writer.writerow([row])
    csvfile.close()

runtime = time.time() - startruntimer
print("runtime: " + str(runtime))


#!/usr/bin/env python
# coding: utf-8
import sys
import pandas as pd
import pickle
import zlib

import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

#Folder and file paths
FOLDER = './' # Folder with this code
PATHS = {'Dataset':FOLDER+'Datasets/newADIdataset/','Train':FOLDER+'Split/Temporal/train_dataset_','Test':FOLDER+'Split/Temporal/test_dataset_','Model':FOLDER+'Split/Temporal/model_'}

df = pd.read_csv('Files/pace.csv')
dist_map = {}
census = list(df['census_code'])
label = list(df['labels'])
for i in range(len(census)):
    dist_map[census[i]] = label[i]

# ----Split a single collection
def Split(dataset_name):
    # Printing the collection name.
    collection_name = dataset_name[8:]
    print('\nCollection:',collection_name.capitalize())

    # Loading the dataset and the model from the drive.
    file = open(PATHS['Dataset']+dataset_name, 'rb')
    dataset = pickle.loads(zlib.decompress(pickle.load(file)))
    file.close()

    df = pd.DataFrame(dataset)
    df.columns = ['ArticleId','Title','Text','Keywords','Date','DistrictId','Emp','Growth','Type']


    Year = 2018 # Year chosen for split
    test_df, train_df = [x for _, x in df.groupby((df['Date'].str[0]).apply(pd.to_numeric) < Year)]

    print(len(train_df))
    print(len(test_df))

    train_df = train_df.values.tolist()
    test_df = test_df.values.tolist()
    file_train = open(PATHS['Train']+collection_name,'wb')
    pickle.dump(zlib.compress(pickle.dumps(train_df),pickle.HIGHEST_PROTOCOL),file_train,pickle.HIGHEST_PROTOCOL)
    file_train.close()

    file_test = open(PATHS['Test']+collection_name,'wb')
    pickle.dump(zlib.compress(pickle.dumps(test_df),pickle.HIGHEST_PROTOCOL),file_test,pickle.HIGHEST_PROTOCOL)
    file_test.close()

# -- Running model training
def DT2V_train(collection_name):

    print(collection_name.capitalize())

    # Loading train dataset
    file_train = open(PATHS['Train']+collection_name,'rb')
    dataset = pickle.loads(zlib.decompress(pickle.load(file_train)))
    file_train.close()

    # Creating the documents with required tags
    documents = [TaggedDocument(i[3],[i[0],i[5],i[6],i[7]]) for i in dataset]
    print('Documents Collected.')

    # Declaring the DT2V Model.
    model = Doc2Vec(vector_size=50,window=3,min_count=3,alpha=0.1,min_alpha=0.001)
    print('Model Initialized.')

    # Building vocabulary.
    model.build_vocab(documents)
    print('Vocabulary size: ',len(model.wv.vocab.keys()))

    # Training model.
    start = time.time()
    for epoch in range(1,101):
        model.train(documents,total_examples=len(documents),epochs=1)
        if epoch==1 or epoch%10==0:
            print('Epoch :',epoch, cosine_similarity([model.docvecs[94],model.docvecs[519]])[0][1])
            print('Elasped Time: ',time.time()-start)
    print('Model Trained.')

    # Saving model.
    file = open(PATHS['Model']+collection_name,'wb')
    model.save(file)
    file.close()
    print('Model Saved.\n\n')

#----------- Methods end---------------#

#------------- Main-------------------#
# Create dataset


# Collect argument and specify the collection
collec = sys.argv[1]

if collec == 'a':
    dataset = 'dataset_agriculture'
elif collec == 'd':
    dataset = 'dataset_development'
elif collec == 'e':
    dataset = 'dataset_environment'
elif collec == 'i':
    dataset = 'dataset_industrialization'
elif collec == 'l':
    dataset = 'dataset_lifestyle'
else:
    dataset = None
    print('dataset not specified')
    exit(0)


SPLIT = sys.argv[2]
TRAIN = sys.argv[3]


# Split the dataset
if SPLIT=='1':
    start = time.time()
    Split(dataset)
    end = time.time()
    print('Time to modify: ',end-start)

# Train the model
if TRAIN=='1':
    start = time.time()
    DT2V_train(str(dataset[8:]))
    end = time.time()
    print('Time to train: ',end-start)

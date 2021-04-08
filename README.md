# Exploring the Scope of Using News Articles to Understand Development Patterns of Districts in India

### Overview

Understanding what factors bring about socio-economic development may often suffer from the streetlight effect, of analyzing the effect of only those variables that have been measured and are therefore available for analysis. How do we check whether all worthwhile variables have been instrumented and considered when building an econometric development model? We attempt to address this question by building unsupervised learning methods to identify and rank news articles about diverse events occurring in different districts of India, that can provide insights about what may have transpired in the districts. This can help determine whether variables related to these events are indeed available or not to model the development of these districts. We also describe several other applications that emerge from this approach, such as to use news articles to understand why pairs of districts that may have had similar socio-economic indicators approximately ten years back ended up at different levels of development currently, and another application that generates a newsfeed of unusual news articles that do not conform to news articles about typical districts with a similar socio-economic profile. These applications outline the need for qualitative data to augment models based on quantitative data, and are meant to open up research on new ways to mine information from unstructured qualitative data to understand development.


### Project Requirements

- Python3.7+
- Following packages needs to be installed to run the scripts:-
    - nltk
    - gensim
    - sklearn
    - scipy
    - numpy
    - pandas
    - pickle
    - zlib
    - matplotlib
    - pymongo
    
    
### Dataset

- Mass media dataset :- Created using crawling 5 online English news sources and categorising them into 5 categories i.e. Agriculture, Development, Environment, Industrialization, Lifestyle along with processing of named entity recognition using OpenCalais.
- Pace of growth labels :- Computed using ADI values of 2011 and 2019 calculated using Census data.
- Employment labels :- Generated using discretization of variables using Census data.
- Google drive link for the dataset used for this work: 
https://drive.google.com/file/d/189FK00Q81IQ5u8EKRkeZVf6vwyzHRG7t/view?usp=sharing

### DocTag2Vec Model

Google drive link for the doctag2vec models for each collection containing vector embeddings for this work:
https://drive.google.com/file/d/18ByxbHogHp5_vOG4FMa6Sgsuq9TEDZ_O/view?usp=sharing
 

### Implementation Details

The brief description about the steps of execution in order along with details of scripts implementation is given as following:-

1. **Dataset Creation**
    - *filterArticles.ipynb* :- Filter out the articles based on keyword based search from the entire crawled corpus of data to create 5 categories.
    - *Articles_Collection.ipynb* :- Maps the articles in collections to their corresponding district ids and filter out the id, title, text, location, published date fields to create dataset for further analysis. 
    - *Dataset_Creation.ipynb* :- Performs location mapping from 2011 to 2001 district ids, pace of growth label mapping, employment label mapping, industry type mapping, entity blinding to create dataset in form of (id, title, text, district id, employment label, industry type, pace of growth label).

2. **Clustering over Vector Embeddings**
    - *trainModel.ipynb* :- Updates the dataset based on NewADI values and trains DocTag2Vec model using Tagged Documents created with (id, district id, employment label, pace of growth label) as tags associated with each document.
    - *vectorClustering.ipynb* :- Performs agglomerative hierarchical clustering over document embeddings computed using DocTag2Vec model where number of clusters are obtained using cophenetic distance obtained from dendrogram visualization. It also removes the outlier clusters based on cophenetic correlation coefficient and selects articles based on proposed ranking method. 

3. **Topic Modeling**
    - *topicModeling.ipynb* :- Performs topic modeling using LDA(Latent Dirichlet Allocation) technique where number of topics is decided using coherence score. 

4. **TFIDF based selection**
    - *tfidfSelection.ipynb* :- Performs selection of articles based on TFIDF aggregate score ranking. 

5. **Evaluation**
    - *quantMetrics.ipynb* :- Computes quantitative metrics i.e. Eucledian distance, Global centroid similarity, Tf-idf based jaccard similarity, Entropy over the articles selected using clustering over vector embeddings, topic modeling and tfidf based selection.
    - *qualMetric.ipynb* :- Computes qualitative metric i.e. t-test and p-score values for review ratings.

6. **Applications**
    - *districtAnalysis.ipynb* :- Selects articles from districts using clustering over vector embeddings based on ranking method.
    - *tSNE.ipynb* :- Visualizes TSNE embeddings for the district vectors for each sub-class. 
    - *unusualTopics.ipynb* :- Selects the articles with highest similarity to global centroid to extract unusual topics.

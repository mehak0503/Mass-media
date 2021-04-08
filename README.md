# Understanding the growth pattern in districts of India using mass media data

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
 

### Implementation Details

The brief description about the steps of execution in order along with details of scripts implementation is given as following:-

1. **Dataset Creation**
    - *filterArticles.ipynb* :- Filter out the articles based on keyword based search from the entire crawled corpus of data to create 5 categories.
    - *Articles_Collection.ipynb* :- Maps the articles in collections to their corresponding district ids and filter out the id, title, text, location, published date fields to create dataset for further analysis. 
    - *Dataset_Creation.ipynb* :- Performs location mapping from 2011 to 2001 district ids, pace of growth label mapping, employment label mapping, industry type mapping, entity blinding to create dataset in form of (id, title, text, district id, employment label, industry type, pace of growth label).

2. **Clustering over Vector Embeddings**
    - *trainModel.ipynb* :- Updates the dataset based on NewADI values and trains DocTag2Vec model using Tagged Documents created with (id, district id, employment label, pace of growth label) as tags associated with each document.
    - *vectorClustering.ipynb* :- 

3. **Topic Modeling**
    - *topicModeling.ipynb* :-

4. **TFIDF based selection**
    - *tfidfSelection.ipynb* :-

5. **Evaluation**
    - *quantMetrics.ipynb* :-
    - *qualMetric.ipynb* :-

6. **Applications**
    - *districtAnalysis.ipynb* :-
    - *tSNE.ipynb* :-
    - *unusualTopics.ipynb* :-

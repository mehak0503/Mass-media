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
    
 

### Implementation Details

The brief description about the steps of execution in order along with details of scripts implementation is given as following:-

1. Dataset Creation

2. Clustering over Vector Embeddings

3. Topic Modeling

4. TFIDF based selection

5. Evaluation

6. Applications


## COMP8240 Application of Data Science : Group K Project

Reimplementation of our paper titled - "Deep Learning for Hate Speech Detection" (to appear in WWW'17 proceedings).

### Requirements
* Pandas
* Tweepy
* JSON
* Keras
* Tensorflow or Theano (we experimented with theano)
* Gensim
* xgboost
* NLTK
* Sklearn
* Numpy

### Dataset:

Source data can be downloaded from https://github.com/zeerakw/hatespeech. 
It contains tweet id's and corresponding annotations where tweets are labelled as either Racist, Sexist or Neither Racist or Sexist.

Twitter API(statuses_lookup) was used to retrieve the tweets using given tweet IDs and some basic processing was done as required by the source code.

Overview:
* Total tweet ids downloaded from source: 17005  
Racism : 2068  
Sexism : 3378  
Neither: 11559  

* Total tweets queried from twitter with above tweet IDs: 10643  
Racism : 76  
Sexism : 2779  
Neither: 11559  

- **Note**: Reduction in number of queried tweets occurred because some of the tweet IDS are no longer valid as their users are not active.

New Data was queried from twitter using the search functionality of twitter API using most some similar words observed in the original work. The retrieved tweets are manually annotated as either Racist, Sexist or Neither Racist or Sexist.

Also, data_collection.ipynb consists of all the codes used  for data collection which includes both New and Orginal data.

- **Note**: To use new data instead of old data change flag =”new” at line 8 in data_handler.py


### Debugging and fixing issues:
1.	Replacement and fixing of several functions due to the python version change from 2.7.x to 3.7.x.
2.	Replacement of deprecated functions.
For example: gensim.models.Word2Vec.load_word2vec_format got deprecated and instead gensim.models.KeyedVectors.load_word2vec_format is used.
3.	Source code readme mentions generation of a model file but unable to identify related files or code. Therefore, new changes were implemented for the same.

### Reimplementing experiments of original work:

Experiments are divided into three parts:

#### PART A: Baseline Approach

```
python BoWV.py --model gradient_boosting --seed 42 -f glove.twitter.27b.25d.txt -d 25 --seed 42 --folds 10 --tokenizer glove --estimators 10 --loss deviance
```
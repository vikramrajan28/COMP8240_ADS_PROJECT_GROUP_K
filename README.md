## COMP8240 Application of Data Science : Group K Project

Reimplementation of our paper titled - "Deep Learning for Hate Speech Detection" (to appear in WWW'17 proceedings).

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

Note: Reduction in number of queried tweets occurred because some of the tweet IDS are no longer valid as their users are not active.

New Data was queried from twitter using the search functionality of twitter API using most some similar words observed in the original work. The retrieved tweets are manually annotated as either Racist, Sexist or Neither Racist or Sexist.


### Debugging and fixing issues:
1.	Replacement and fixing of several functions due to the python version change from 2.7.x to 3.7.x.
2.	Replacement of deprecated functions.
For example: gensim.models.Word2Vec.load_word2vec_format got deprecated and instead gensim.models.KeyedVectors.load_word2vec_format is used.
3.	Source code readme mentions generation of a model file but unable to identify related files or code. Therefore, new changes were implemented for the same.

### Reimplementing experiments of original work:

Divided into three parts:

#### PART A: Baseline Approach

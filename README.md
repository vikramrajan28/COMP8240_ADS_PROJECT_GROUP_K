## COMP8240 Application of Data Science : Group K Project

Reimplementation of the paper titled - "Deep Learning for Hate Speech Detection" (to appear in WWW'17 proceedings).

By 
* Bhushan Samarth (45818592)  
* Aakash Sadaphule (45817901)
* Vikram Rajan (45763054)
* Vikram Mhaskey (45714819)

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

Experiments are divided into three parts and instructions to run the code is attached below:
Note: For further details of the parameters of different model(CNN,BoW,lstm,etc) python files please refer to SourceCode_readme.md.

#### PART A: Baseline Approach

1. Bag of Words Vector + Gradient boosting classifier
```
python BoWV.py --model gradient_boosting --seed 42 -f glove.twitter.27b.25d.txt -d 25 --seed 42 --folds 10 --tokenizer glove --estimators 10 --loss deviance
```
2. Bag of Words Vector + Balanced SVM
```
python BoWV.py --model svm --seed 42 -f glove.twitter.27b.25d.txt -d 25 --seed 42 --folds 10 --tokenizer glove --kernel rbf -- class_weight balanced
```
3. TFIDF + Gradient boosting classifier:
```
python tfidf.py -m tfidf_gradient_boosting --max_ngram 3 --tokenizer glove --loss deviance --estimators 10
``` 
4. TFIDF + Balanced SVM:
```
python tfidf.py -m tfidf_svm --max_ngram 3 --tokenizer glove --loss deviance  --class_weight balanced --kernel rbf
```

#### PART B: Proposed Approach
1. CNN + Random embeddings
```
python cnn.py -f C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt -d 25 --tokenizer nm --loss categorical_crossentropy --optimizer adam --epochs 10 --batch-size 128 --initialize-weights random --scale-loss-function
```
2. CNN + GloVe embeddings
```
python cnn.py -f C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt -d 25 --tokenizer nm --loss categorical_crossentropy --optimizer adam --epochs 10 --batch-size 128 --initialize-weights glove --scale-loss-function
```
3. FastText + GloVe embeddings 
```
python fast_text.py 25 glove
```
4. FastText + random embeddings 
```
python fast_text.py 25 random
```
5. LSTM + random embeddings 
```
python lstm.py -f C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt -d 25 --tokenizer nm --loss categorical_crossentropy --optimizer adam --initialize-weights random --learn-embeddings --epochs 10 --batch-size 512
```
6. LSTM + GloVe embeddings 
```
python lstm.py -f C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt -d 25 --tokenizer nm --loss categorical_crossentropy --optimizer adam --initialize-weights glove --learn-embeddings --epochs 10 --batch-size 512
```

#### PART C:  Embeddings learned from NN to classifiers
1.	Fasttext + random + GDBT

Step 1: creating/saving model and vocabulary files
```
python fast_text.py 25 random
```
Step 2: Training the above saved model files on NN classifier
```
python nn_classifier.py C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt 25 gradient_boosting fast_text.npy vocab_fast_text
```

2. 	FastText + Glove + GDBT

Step 1: creating/saving model and vocabulary files
```
python fast_text.py 25 glove
```
Step 2: Training the above saved model files on NN classifier
```
python nn_classifier.py C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt 25 gradient_boosting fast_text.npy vocab_fast_text
```

3.	CNN + glove + GBDT:

Step 1: creating/saving model and vocabulary files
```
python cnn.py -f C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt -d 25 --tokenizer nm --loss categorical_crossentropy --optimizer adam --epochs 10 --batch-size 128 --initialize-weights glove --scale-loss-function
```
Step 2: Training the  above saved model files on NN classifier
```
python nn_classifier.py C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt 25 gradient_boosting cnn.npy vocab_cnn
```

4.	CNN + random + GBDT:

Step 1: creating/saving model and vocabulary files
```
python cnn.py -f C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt -d 25 --tokenizer nm --loss categorical_crossentropy --optimizer adam --epochs 10 --batch-size 128 --initialize-weights random --scale-loss-function
```
Step 2: Training the above saved model files on NN classifier
```
python nn_classifier.py C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt 25 gradient_boosting cnn.npy vocab_cnn
```

5.	LSTM + random + GBDT:

Step 1: creating/saving model and vocabulary files
```
python lstm.py -f C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt -d 25 --tokenizer nm --loss categorical_crossentropy --optimizer adam --initialize-weights random --learn-embeddings --epochs 10 --batch-size 512
```
Step 2: Training the above saved model files on NN classifier
```
python nn_classifier.py C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt 25 gradient_boosting lstm.npy vocab_lstm
```

6.	LSTM + GloVe + GBDT:

Step 1: creating/saving model and vocabulary files
```
python lstm.py -f C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt -d 25 --tokenizer nm --loss categorical_crossentropy --optimizer adam --initialize-weights glove --learn-embeddings --epochs 10 --batch-size 512
```
Step 2: Training the above saved model files on NN classifier
```
python nn_classifier.py C:/My_Workspace/Git/ADS_Project/glove.twitter.27B.25d.txt 25 gradient_boosting lstm.npy vocab_lstm
```

** Note: Repeat same instructions above for new data.

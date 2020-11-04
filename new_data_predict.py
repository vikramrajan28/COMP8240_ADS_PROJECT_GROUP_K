# Changes for new data prediction by Vikram Rrjan

from collections import defaultdict
import pandas as pd
import numpy as np
import pdb
from string import punctuation
from preprocess_twitter import tokenize as Tokenize
from gensim.parsing.preprocessing import STOPWORDS
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import  f1_score, accuracy_score, recall_score, precision_score


nvocab, reverse_nvocab = {}, {}
nfreq = defaultdict(int)
twts = {}
def newTweetsPred(weight, model,MAX_SEQUENCE_LENGTH,word2vec_model):
    print("\n\n#### Predicting for New Data #####",len(X_n))

    twts = select_new_tweets(word2vec_model)
    #pdb.set_trace()
    gen_nvocab()
    X_n, y_n = gen_nsequence(twts)
    print("Total tweets to predict: ",len(X_n))
    #print("max seq length is %d"%(MAX_SEQUENCE_LENGTH))
    data = pad_sequences(X_n, maxlen=MAX_SEQUENCE_LENGTH)
    y_n = np.array(y_n)
    model.layers[0].set_weights([weight])
    y_npred = model.predict(data)
    y_npred = np.argmax(y_npred, axis=1)
    print("Weighted Precision Score", precision_score(y_n, y_npred, average='weighted'))
    print("Micro Precision Score",precision_score(y_n, y_npred, average='micro'))
    print("Weighted recall Score",recall_score(y_n, y_npred, average='weighted'))
    print("Micro recall Score",recall_score(y_n, y_npred, average='micro'))
    print("Weighted F1 Score",f1_score(y_n, y_npred, average='weighted'))
    print("Micro F1 Score", f1_score(y_n, y_npred, average='micro'))
    print("\n#### End of Prediction #####",len(X_n))
    #pdb.set_trace()

def get_ndata():
    ts = []
    file_name = 'C:/Users/Owner/Downloads/FINAL_ASs_DF.xlsx'
    df = pd.read_excel(file_name, sheet_name = "Sheet2")
    for i,v in df.iterrows():
        ts.append({
                'id': v['id'],
                'text': str(v['text']).lower(),
                'label': v['Column2']
                })
    #pdb.set_trace()
    print("Total new tweets: ",len(ts))
    return ts
    
def select_new_tweets(word2vec_model):
    # selects the tweets as in mean_glove_embedding method
    # Processing
    twts = get_ndata()
    twt_return = []
    for twt in twts:
        _emb = 0
        wds = Tokenize(twt['text']).split()
        for w in wds:
            if w in word2vec_model:  # Check if embeeding there in GLove model
                _emb+=1
        if _emb:   # Not a blank tweet
            twt_return.append(twt)
    print ('Tweets selected:', len(twt_return))
    #pdb.set_trace()
    return twt_return

def gen_nvocab():
    # Processing
    vocab_index = 1
    for tweet in twts:
        text = Tokenize(tweet['text'])
        text = ''.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]

        for word in words:
            if word not in nvocab:
                nvocab[word] = vocab_index
                reverse_nvocab[vocab_index] = word       # generate reverse vocab as well
                vocab_index += 1
            nfreq[word] += 1
    nvocab['UNK'] = len(nvocab) + 1
    reverse_nvocab[len(nvocab)] = 'UNK'
    #pdb.set_trace()

def gen_nsequence(t):
    twts=t
    y_map = {
            'neither': 0,
            'racism': 1,
            'sexism': 2
            }

    X, y = [], []
    for tweet in twts:
        text = Tokenize(tweet['text'])
        text = ''.join([c for c in text if c not in punctuation])
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        seq, _emb = [], []
        for word in words:
            seq.append(nvocab.get(word, nvocab['UNK']))
        X.append(seq)
        y.append(y_map[tweet['label']])
    return X, y
# Changes for new data prediction by Vikram R

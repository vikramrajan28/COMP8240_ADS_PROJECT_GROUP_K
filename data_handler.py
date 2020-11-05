import json
import pdb
import codecs
import pdb
import pandas as pd

# Changes for new data prediction by Vikram Rrjan
flg="new" # Set Flag = new, if using new data instead of orginal data

def get_data():
    tweets = []
    # Changes for new data prediction by Vikram Rrjan
    if(flg=="new"):
        file_name = './tweet_data/NewData/tweets.xlsx'
        df = pd.read_excel(file_name, sheet_name = "Sheet1")
        for i,v in df.iterrows():
            tweets.append({
                    'id': v['id'],
                    'text': str(v['text']).lower(),
                    'label': v['labels']
                    })
    else:    
        files = ['racism.json', 'neither.json', 'sexism.json']
        for file in files:
            with codecs.open('./tweet_data/' + file, 'r', encoding='utf-8') as f:
                data = f.read()
                tweet_full = json.loads(data)
            for line in tweet_full:
                #print(line)
                #tweet_full = json.loads(line)
                tweets.append({
                    'id': line['id'],
                    'text': line['text'].lower(),
                    'label': line['Annotation'],
                    'name': line['user']['name'].split()[0]
                    })

    #pdb.set_trace()
    print("Total tweets: ",len(tweets))
    return tweets


if __name__=="__main__":
    tweets = get_data()
    males, females = {}, {}
    with open('./tweet_data/males.txt') as f:
        males = set([w.strip() for w in f.readlines()])
    with open('./tweet_data/females.txt') as f:
        females = set([w.strip() for w in f.readlines()])
    
    males_c, females_c, not_found = 0, 0, 0
    for t in tweets:
        if t['name'] in males:
            males_c += 1
        elif t['name'] in females:
            females_c += 1
        else:
            not_found += 1
    print(males_c, females_c, not_found)
    pdb.set_trace()

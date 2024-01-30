#!/usr/bin/env python
from __future__ import print_function

import numpy as np
import pickle
from collections import defaultdict
import sys, re
import pandas as pd
import json

# noinspection PyCompatibility
from builtins import range

COMMENTS_FILE = "../data/comments.json"
TRAIN_MAP_FILE = "../data/my_train_balanced.csv"
TEST_MAP_FILE = "../data/my_test_balanced.csv"

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data
    """
    revs = []

    sarc_train_file = data_folder[0]
    sarc_test_file = data_folder[1]
    
    train_data = np.asarray(pd.read_csv(sarc_train_file, header=None))
    test_data = np.asarray(pd.read_csv(sarc_test_file, header=None))

    comments = json.loads(open(COMMENTS_FILE).read())
    vocab = defaultdict(float)



    
    for line in train_data: 
       
        label_str = line[2]
        if( label_str == 0):
            label = [1, 0]
        else:
            label = [0, 1]
       
        rev = [comments[line[0]]['text'].strip()]
        posts = [comments[match]["text"].strip() for match in re.findall(r"'([^']*)'", line[1])]
       
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
            posts = clean_str(" ".join(posts))
        else:
            orig_rev = " ".join(rev).lower()

        combined = posts +" "+ orig_rev
        
        datum  = {"y":int(1), 
                  "id":line[0],
                  "comment": orig_rev,
                  "post": posts,
                  "combined": combined ,
                  "author": comments[line[0]]['author'],
                  "topic": comments[line[0]]['subreddit'],
                  "label": label,
                  "num_words": len(combined.split()),
                  "split": int(1)}
    
        revs.append(datum)
        
    for line in test_data:
    
        label_str = line[2]
        if( label_str == 0):
            label = [1, 0]
        else:
            label = [0, 1]
        rev = [comments[line[0]]['text'].strip()]
        posts = [comments[match]["text"].strip() for match in re.findall(r"'([^']*)'", line[1])]
        if clean_string:
            orig_rev = clean_str(" ".join(rev))
            posts = clean_str(" ".join(posts))
        else:
            orig_rev = " ".join(rev).lower()

        combined = posts +" "+ orig_rev
    
        datum  = {"y":int(1),
                  "id": line[0], 
                  "text": orig_rev, 
                  "post": posts,
                  "combined": combined,
                  "author": comments[line[0]]['author'],
                  "topic": comments[line[0]]['subreddit'],
                  "label": label,
                  "num_words": len(combined.split()),                      
                  "split": int(0)}
        revs.append(datum)
        

    return revs


def clean_str(text, case= True):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
   # Replace or remove URLs
    text = re.sub(r'http\S+', '', text)

    # Replace email addresses with a generic word
    text = re.sub(r'\S*@\S*\s?', '[email]', text)

    # Replace newlines and tabs with a space
    text = text.replace('\n', ' ').replace('\t', ' ')
    return text.strip() if case else text.strip().lower()





if __name__=="__main__":  

 
    data_folder = [TRAIN_MAP_FILE,TEST_MAP_FILE] 
    print("loading data...")
    revs = build_data_cv(data_folder,  cv=10, clean_string=True)

    pickle.dump(revs, open("mainbalancedpickle-new.p", "wb"))
    print("dataset created!")
   
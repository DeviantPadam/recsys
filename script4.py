#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 02:02:46 2020

@author: deviantpadam
"""

import argparse
import pandas as pd
import numpy as np
import concurrent.futures
import os
import tqdm
from collections import Counter
from torch2vec.data import DataPreparation
from torch2vec.torch2vec import DM


def get_bigrams(corpus,min_count):
    text = np.copy(corpus)
    vocab = [word for sen in text for word in sen]
    ngram = [(i,j) for i,j in zip(vocab[:-1],vocab[1:])]
    freq = Counter(ngram)
    filterbi = [bigram for bigram in freq.most_common() if bigram[1]>min_count]
    bigrams = [" ".join(bigram[0]) for bigram in filterbi]
    return bigrams

def _add_bigrams(text):
    for idx in range(len(text)):
        length=len(text[idx])-1
        word_count=0
        while word_count<length:

            if text[idx][word_count]+' '+text[idx][word_count+1] in bigrams:
                text[idx][word_count] = text[idx][word_count]+' '+text[idx][word_count+1]
                text[idx].remove(text[idx][word_count+1])
                length = len(text[idx])-1
    #             print(cor[i][j]+' '+cor[i][j+1])

            word_count+=1
    return text    
    

def phraser(corpus,workers=-1):
    if workers==-1:
        workers = os.cpu_count()
    chunks = np.array_split(corpus,workers)
    with concurrent.futures.ProcessPoolExecutor(workers) as executor:
        result = np.concatenate(list(tqdm.tqdm(executor.map(_add_bigrams,chunks),total=workers,desc='Phrasing using {} cores'.format(workers))),axis=0)
        executor.shutdown(wait=True)
#     result = _add_bigrams(data)
    global bigrams
    del bigrams
    return pd.DataFrame({'text':np.array(result)})['text']
    
    
corpus_path = 'recsys/suggest_dump.txt'
save_to = 'weights'


train = pd.read_csv(corpus_path,delimiter='\t')
train.fillna('none',inplace=True)
corpus = train['title']+' '+train['summary']+' '+train['authors']+' '+train['subjects']+' '+train['tasks']
corpus.name='text'
train = pd.concat([train[['id','authors','subjects','tasks']],corpus],axis=1)
train['authors']=(train['authors'].str.lower()).str.split(',')
train['subjects'] = (train['subjects'].str.lower()).str.split(',')
train['tasks']= (train['tasks'].str.lower()).str.split(',')
data = DataPreparation(train,f_size=3)
data.tokenize()
del train, corpus
bigrams = get_bigrams(data.corpus.values,min_count=700)
data.corpus = phraser(data.corpus.values)
bigrams = get_bigrams(data.corpus.values,min_count=500)
data.corpus = phraser(data.corpus.values)
data.vocab_builder()
doc, context, target_noise_ids = data.get_data(window_size=5,num_noise_words=10)

model = DM(vec_dim=100,num_docs=len(data),num_words=data.vocab_size).cuda()

num_workers = os.cpu_count()
model.fit(doc_ids=doc,context=context,target_noise_ids=target_noise_ids,epochs=20,batch_size=8000,num_workers=num_workers)

model.save_model(data.document_ids,data.args,file_name=save_to)
print('Remember this please to use in Load model (pad=[author,subjects,tasks])=',data.pad_length)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 20:53:55 2020

@author: deviantpadam
"""


import pandas as pd
import numpy as np
import concurrent.futures
import os
import tqdm
from collections import Counter
from torch2vec.data import DataPreparation, Dataset
from torch2vec.torch2vec_light import DM
import pytorch_lightning as pl
import torch

# train = pd.read_csv('/home/deviantpadam/Downloads/example.csv',delimiter='\t')
train = pd.read_csv('/home/deviantpadam/Downloads/example (1).csv')
def cleaner(train):
    sub=(train['subjects'].str.lower()).str.split(',',expand=True)
    sub.drop([2,3],axis=1,inplace=True)
    sub.columns = ['subject1','subject2']
    sub.fillna('none',inplace=True)
    tasks = (train['tasks'].str.lower()).str.split(',',expand=True)[0]
    tasks.fillna('none',inplace=True)
    tasks.name = 'task'
    train = pd.concat([train,sub,tasks],axis=1).drop(['subjects','tasks'],axis=1)
    train.fillna('none',inplace=True)
    return train
train = cleaner(train)
corpus = train['authors']+' '+train['title']+' '+train['summary']+' '+train['subject1']+' '+train['subject2']+' '+train['task']
corpus.name = 'text'
corpus = pd.concat([train['subject1'],train['subject2'],train['task'],corpus],axis=1)

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
    
def _get_bigrams(corpus,min_count):
    text = np.copy(corpus)
    vocab = [word for sen in text for word in sen]
    ngram = [(i,j) for i,j in zip(vocab[:-1],vocab[1:])]
    freq = Counter(ngram)
    filterbi = [bigram for bigram in freq.most_common() if bigram[1]>min_count]
    bigrams = [" ".join(bigram[0]) for bigram in filterbi]
    return bigrams
    

data = DataPreparation(corpus.reset_index(),f_size=3)
data.tokenize()
bigrams = _get_bigrams(data.corpus.values,min_count=700)
data.corpus = phraser(data.corpus.values)
bigrams = _get_bigrams(data.corpus.values,min_count=500)
data.corpus = phraser(data.corpus.values)

data.vocab_builder()
doc, context, target_noise_ids = data.get_data(window_size=5,num_noise_words=10)

num_workers = os.cpu_count()
dataset = Dataset(doc, context, target_noise_ids)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=10000
                                                 )


model = DM(vec_dim=100,num_docs=len(data),num_words=data.vocab_size)
trainer = pl.Trainer(gpus=[0],max_epochs=20)
trainer.fit(model,train_dataloader=dataloader)

model.save_model(data.document_ids,data.args,file_name='weights')
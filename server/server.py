#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 23:55:25 2020

@author: deviantpadam
"""

from flask import Flask
from torch2vec.torch2vec import LoadModel

app = Flask(__name__)
model = LoadModel('../train/weights.npy')

@app.route('/')
def main():
    ids, prob = model.similar_docs(124,topk=10,use='torch')
    ids = "------------".join(ids)
    type(ids)
    prob= [str(i) for i in prob]
    prob = "-----------".join(prob)
    return ids+prob

@app.route('/models',methods=['GET','POST'])
def models():
    similarity = None
    results = None
    w = str(request.args.get('type'))
    a = []
    similarity = model.similar_docs(int(w),use='sklearn')

if '__main__'==__name__:
    app.run(host="0.0.0.0", port="5002")

#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from torch2vec.torch2vec import DM
from torch2vec.data import DataPreparation
# pd.read_csv('../input/')

data = DataPreparation('../input/recsysluc/semantic_dump.txt',vocab_size=int(2e5)) #vocab_size to restrict vocabulary size

data.vocab_builder()

doc, context, target_noise_ids = data.get_data(window_size=3,num_noise_words=6)

model = DM(vec_dim=100,num_docs=len(data),num_words=data.vocab_size)

num_workers = os.cpu_count()
model.fit(doc_ids=doc,context=context,target_noise_ids=target_noise_ids,epochs=5,batch_size=3000,num_workers=num_workers)

model.save_model(ids=data.document_ids,file_name='weights')


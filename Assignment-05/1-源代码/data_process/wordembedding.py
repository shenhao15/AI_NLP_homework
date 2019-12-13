# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:28:21 2019

@author: zhangwei
"""
import re
from gensim.models import word2vec
import os
import logging
from collections import Counter
from collections import defaultdict
import numpy as np

pattern_sw=re.compile("(\d+) ([\s\S]+?) ")
list_stopwords=[]
with open('C:/Users/zhangwei/Downloads/sqlResult_1558435/raw_stopwords.txt',mode='r',encoding='utf-8') as sw_file:
    stopwords=sw_file.read()
    processed_sw=pattern_sw.findall(stopwords)
    for word in processed_sw:
        list_stopwords.append(word[1])
    undup_stopwords=list(set(list_stopwords))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
input_folder='C:/Users/zhangwei/Downloads/jiebasplit/jiebasplit'
filenames=os.listdir(input_folder)
sentences=[]
counter_sentences=[]
words_dict=defaultdict()
for file in filenames:
    input_file=open(input_folder+'/'+file,encoding='utf-8')
    print(file)
    for sent in input_file.readlines():
        list_sent=sent[:-1].split(' ')
        for word in list_sent:
            if word in undup_stopwords:
                list_sent.remove(word)
        sentences.append(list_sent)
        counter_sentences+=list_sent
all_words=list(set(counter_sentences))

counter=Counter(counter_sentences)
for word in all_words:
    words_dict[word]=counter[word]/len(counter_sentences)
np.save('C:/Users/zhangwei/Downloads/words_dict.npy',words_dict)        


model_path='C:/Users/zhangwei/Downloads/'
model_skipgram=word2vec.Word2Vec(sentences,min_count=5,iter=20,size=200,window=5,sg=1,negative=5)
model_skipgram.wv.save_word2vec_format(model_path+'/model_skipgram.model',binary=False)

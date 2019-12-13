# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:49:49 2019

@author: zwconnie
"""
import re
import os
import jieba

words=[]
simp_folder='E:/AI/NLP/Project1/wikiextractor-master/simplified'
filenames=os.listdir(simp_folder)
jieba_folder='E:/AI/NLP/Project1/wikiextractor-master/jiebasplit'
#simp_folder='E:/AI/NLP/Project1'
#filenames=['test.txt']
para_pattern=re.compile('''<doc id([\S\s]+?)>[\n']([\u4E00-\u9FA5\w]+?)[\n]{2}([\S\s]+?)</doc>''')
for filename in filenames:
    input_file=open(simp_folder+'/'+filename,encoding='utf-8')
    with open(jieba_folder+'/jieba_'+filename+'.txt',mode='a',encoding='utf-8') as output_file:
        paras=input_file.read()
        for item in para_pattern.findall(paras): 
            jieba.add_word(item[1])
            content=item[2]
            for line in content.split('\n'):
                if not line.strip():continue
                line=line.replace('[','').replace('(','').replace('（','').replace(')','').replace('）','').replace('{','').replace('}','').replace(']','').replace('《','').replace('》','').replace('「','').replace('」','').replace('『','').replace('』','').replace('“','').replace('”','').replace('‘','').replace('’','')
                line=re.sub('(\s+)','',line)
                #line_replaced=re.sub('''([(（)）{}[]《》「」『』“”‘’\s]+)''','',line)
                output_file.write(' '.join(jieba.cut(line))+'\n')
            
    
  
    


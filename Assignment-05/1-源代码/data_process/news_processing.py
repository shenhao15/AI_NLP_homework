# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 19:16:48 2019

@author: zhangwei
"""
import re
import jieba
import pandas as pd
import numpy as np

df_input=pd.read_csv('C:/Users/zhangwei/Downloads/sqlResult_1558435/sqlResult_1558435.csv',encoding='gb18030')
df_title=df_input[['id','title']]
sum_pattern=re.compile('''summary":"([\s\S]+?)"''')
summary=''
with open('C:/Users/zhangwei/Downloads/jiebasplit/jiebasplit/jieba_swiki_13.txt',mode='a',encoding='utf-8') as output_file:
    for i in range(len(df_input.feature)):
        if len(sum_pattern.findall(df_input.loc[i,'feature']))==0:
            summary=df_input.loc[i,'content']
        elif pd.isnull(df_input.loc[i,'content']):
            summary=sum_pattern.findall(df_input.loc[i,'feature'])[0]
        else:
            summary=sum_pattern.findall(df_input.loc[i,'feature'])[0]+df_input.loc[i,'content']
        if pd.isnull(summary):
            df_title.loc[i,'paragraph']=np.nan
        else:
            summary=summary.replace(' ','').replace('【','').replace('】','').replace('\u3000','').replace('\\r','').replace('\\n','').replace('[','').replace('(','').replace('（','').replace(')','').replace('）','').replace('{','').replace('}','').replace(']','').replace('《','').replace('》','').replace('「','').replace('」','').replace('『','').replace('』','').replace('“','').replace('”','').replace('‘','').replace('’','')
            summary=' '.join(jieba.cut(summary))
            df_title.loc[i,'paragraph']=summary
            output_file.write(summary+'\n')

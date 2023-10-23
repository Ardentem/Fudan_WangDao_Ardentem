import numpy as np
import torch 
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
from tqdm import tqdm
import pandas as pd
 
class basicmodel:
    def __init__(self,modelselect='bert_cn'):
        #MODEL_PATH = './Transformer_Bert/bert_base_chinese'
        #设置预训练模型的位置
        self.MODEL_PATH = './Transformer_Bert/bert_base_chinese'
        if modelselect == 'finbert':
            self.MODEL_PATH = './Transformer_Bert/FinBERT_L-12_H-768_A-12_pytorch'
        #股票收盘价 读取
        self.close = pd.read_feather('./database/BasicFactor_Close.txt').set_index('time')
        self.close.index = pd.to_datetime(self.close.index.astype(str))
        self.close = self.close['2022-06-01':]

    def initialize(self):
        # a. 通过词典导入分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_PATH)
        self.tokenizer.model_max_length=512
        # b. 导入配置文件
        self.model_config = BertConfig.from_pretrained(self.MODEL_PATH)
        # 修改配置
        self.model_config.output_hidden_states = True
        self.model_config.output_attentions = True
        # 通过配置和路径导入模型
        self.bert_model = BertModel.from_pretrained(self.MODEL_PATH, config = self.model_config)

    # 定义计算相似度的函数（两个字符串，单独调用使用，不要在for循环中大量调用，速度很慢）
    def calc_similarity(self,s1,s2):
        # 对句子进行分词，并添加特殊标记（开始和结尾的标记）
        inputs = self.tokenizer([s1, s2], return_tensors='pt', padding=True, truncation=True)
        # 将输入传递给BERT模型，并获取输出
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            #此函数调用非常慢
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # 计算余弦相似度，并返回结果
        sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return sim
    
    #句子list bert模型处理（可以处理很大量句子list）
    def sentece_bert(self,sentencelist):
        # 对句子进行分词，并添加特殊标记
        n = len(sentencelist)
        # 句子长度20个一算，否则内存会爆
        reslist = []
        for i in tqdm(range(1+int( (n-1) / 20 ))):
            inputs = self.tokenizer(sentencelist[20*i:20*(i+1)], return_tensors='pt', padding=True, truncation=True)
            # 将输入传递给BERT模型，并获取输出
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            reslist.extend(list(embeddings))
        return reslist
    
    #导入股票信息
    def initialize_stockinfo(self,modelselect='bert_cn'):
        #股票info 句子向量数据读取
        self.stockinfo_vec = pd.read_feather('./database/infoall_vec_BasicCN.txt')
        if modelselect == 'finbert':
            self.stockinfo_vec = pd.read_feather('./database/infoall_vec_FinCN.txt')
    
    #查找当日关于该信息最相关的股票
    def corr_max_stock(self,info_vec,date,k,info_name='introduction'):
        #info_name: code, shortname, name, introduction
        #如果当天没close数据，则这个股票的信息不参与排名
        dayinfo = self.stockinfo_vec[info_name][self.close.loc[date].reset_index().dropna().index]
        simlist = []
        for stockinfo in dayinfo.dropna().items():
            #计算余弦相似度
            sim = np.dot(info_vec,stockinfo[1]) / (np.linalg.norm(info_vec) * np.linalg.norm(stockinfo[1]))
            simlist.append((stockinfo[0],sim))
        #每一对数据集的第一个数字是在股票信息df的index序号，第二个是相似度
        simlist.sort(key=lambda x:x[1])
        #返回相似度最高的k支股票
        return(simlist[-k:])
    
    #股票收益数据处理
    def initialize_stockret(self):
        self.adjfac = pd.read_feather('./database/BasicFactor_AdjFactor.txt').set_index('time')
        self.adjfac.index = pd.to_datetime(self.adjfac.index.astype(str))
        self.adjfac = self.adjfac['2022-06-01':]
        self.adjclose = self.close * self.adjfac
        self.ret = self.adjclose / self.adjclose.shift(1) -1



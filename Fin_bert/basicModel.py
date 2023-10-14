import numpy as np
import torch 
from transformers import BertTokenizer, BertConfig, BertForMaskedLM, BertForNextSentencePrediction
from transformers import BertModel
 
class basicmodel:
    def __init__(self):
        #MODEL_PATH = './Transformer_Bert/bert_base_chinese'
        #设置预训练模型的位置
        self.MODEL_PATH = './Transformer_Bert/FinBERT_L-12_H-768_A-12_pytorch'

    def initialize(self):
        # a. 通过词典导入分词器
        self.tokenizer = BertTokenizer.from_pretrained(self.MODEL_PATH)
        # b. 导入配置文件
        self.model_config = BertConfig.from_pretrained(self.MODEL_PATH)
        # 修改配置
        self.model_config.output_hidden_states = True
        self.model_config.output_attentions = True
        # 通过配置和路径导入模型
        self.bert_model = BertModel.from_pretrained(self.MODEL_PATH, config = self.model_config)

    # 定义计算相似度的函数
    def calc_similarity(self,s1,s2):
        # 对句子进行分词，并添加特殊标记
        inputs = self.tokenizer([s1, s2], return_tensors='pt', padding=True, truncation=True)
        # 将输入传递给BERT模型，并获取输出
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        # 计算余弦相似度，并返回结果
        sim = np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return sim



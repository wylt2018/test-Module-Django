import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel


class Config(object):
    def __init__(self):
        self.model_name = 'FirstClassifier'
        self.save_path = '/save_dict/' + self.model_name + '.ckpt'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_epochs = 3
        self.batch_size = 4
        self.pad_size = 512
        self.learning_rate = 1e-5
        self.bert_path = 'bert_pretrained'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


class Model(nn.Module):
    # 此处的embedding是embedding matrix
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)  # 加载预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True  # 代表做微调
        self.fc = nn.Linear(config.hidden_size, 2)

    def forward(self, x, extract=False):  # 前向传播过程
        # 输入内容x:[ids,seq_len,mask]
        if extract:
            context = x[0]
            content = context[0].reshape(4, 1)
            for i in range(1, len(context)):
                temp = context[i].reshape(4, 1)
                content = torch.cat((content, temp), dim=1)
            mask = x[2]
            mask1 = mask[0].reshape(4, 1)
            for i in range(1, len(mask)):
                temp = mask[i].reshape(4, 1)
                mask1 = torch.cat((mask1, temp), dim=1)
            _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
            out = self.fc(pooled)
            return out
        else:
            context = x[0]
            context = torch.LongTensor(context)
            mask = torch.ones(5)
            word2vec, _ = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
            return word2vec
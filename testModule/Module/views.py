import json
from django.shortcuts import render
# Create your views here.
from bs4 import BeautifulSoup
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
import torch.nn as nn
from django.http import HttpResponse
bert_path = 'bert_pretrained'
result = 100


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained('Module/bert_pretrained')  # 加载预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True  # 代表做微调
        self.fc = nn.Linear(768, 2)

    def forward(self, x):
        context = x[0]
        mask = x[1]
        context = torch.unsqueeze(context, 0)
        mask = torch.unsqueeze(mask, 0)
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        print(out)
        return out


def word_clean(sentence):
    content = BeautifulSoup(sentence, "html.parser").get_text()
    content = content.lower()
    tokenizer = BertTokenizer.from_pretrained('Module/bert_pretrained')
    token = tokenizer.tokenize(content)
    token.insert(0, '[CLS]')
    if len(token) < 512:
        mask = [1] * len(token) + [0] * (512 - len(token))
        token.insert(len(token), '[SEP]')
        token_ids = tokenizer.convert_tokens_to_ids(token)
        token_ids = token_ids + ([0] * (512 - len(token)))
        seq_len = len(token)
    else:
        mask = [1] * 512
        token.insert(511, '[SEP]')
        token = token[:512]
        token_ids = tokenizer.convert_tokens_to_ids(token)
        seq_len = 512
    token_ids = torch.LongTensor(token_ids)
    mask = torch.LongTensor(mask)
    return token_ids, seq_len, mask


def predict(model, context):
    model.eval()
    with torch.no_grad():
        outputs = model(context)
    return outputs


def index(request):
    global result
    if request.method == "POST":
        sentence = request.POST.get('sentence')
        token_ids, _, mask = word_clean(sentence)
        context = (token_ids, mask)
        model = Model()
        output = predict(model=model, context=context)
        answer = int(torch.max(output.data, 1)[1])
        print(output)
        print(answer)
        result = {"result": answer, "msg": "correct"}
        return HttpResponse(json.dumps(result, ensure_ascii=False))
    return render(request, 'Module/index.html')

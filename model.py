import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
import json

#transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from transformers import BertModel
from transformers import AutoTokenizer, AutoModel
from transformers import ElectraModel, ElectraTokenizer

from sklearn.utils.class_weight import compute_class_weight


class sentencEmojiDataset(Dataset):
    def __init__(self, directory, tokenizer):
        data = pd.read_csv(directory, encoding='UTF-8')

        self.tokenizer = tokenizer
        self.sentences = list(data.iloc[:, 0])

        emojis = list(data.iloc[:, 1])
        emojis_unique = list(set(emojis))

        self.labels = [emojis_unique.index(i) for i in emojis]

        self.labels_dict = {'key': range(len(emojis_unique)), 'value': emojis_unique}

    def __getitem__(self, i):  # collate 이전 미리 tokenize를 시켜주자
        tokenized = self.tokenizer(str(self.sentences[i]), return_tensors='pt')

        # 아래 세 개는 tokenizer가 기본적으로 반환하는 정보. BERT의 input이기도 함
        input_ids = tokenized['input_ids']
        token_type_ids = tokenized['token_type_ids']
        attention_mask = tokenized['attention_mask']

        #         print(str(self.sentences[i]) +' : ')
        #         print(tokenized)

        return {'input_ids': input_ids, 'token_type_ids': token_type_ids,
                'attention_mask': attention_mask, 'label': self.labels[i]}

    def __len__(self):  # data loader가 필요로 하여 필수적으로 있어야 하는 함수
        return len(self.sentences)


class collate_fn:
    def __init__(self, labels_dict):
        self.num_labels = len(labels_dict)

    def __call__(self, batch):  # batch는 dataset.getitem의 return 값의 List. eg. [{}, {}. ...]
        # batch내 최대 문장 길이(토큰 개수)를 먼저 구해서 padding할 수 있도록 하기
        batchlen = [sample['input_ids'].size(1) for sample in batch]  # tensor값을 반환하기 때문에 1번째 차원의 길이를 구함
        maxlen = max(batchlen)
        input_ids = []
        token_type_ids = []
        attention_mask = []
        # padding: [5, 6] [0, 0,  ...]을 concatenate 하는 방식으로 패딩
        for sample in batch:
            pad_len = maxlen - sample['input_ids'].size(1)
            pad = torch.zeros((1, pad_len), dtype=torch.int)
            input_ids.append(torch.cat([sample['input_ids'], pad], dim=1))
            token_type_ids.append(torch.cat([sample['token_type_ids'], pad], dim=1))
            attention_mask.append(torch.cat([sample['attention_mask'], pad], dim=1))
        # batch 구성
        input_ids = torch.cat(input_ids, dim=0)
        token_type_ids = torch.cat(token_type_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        # one-hot encoding
        # batch 내 라벨을 tensor로 변환
        tensor_label = torch.tensor([sample['label'] for sample in batch])

        return input_ids, token_type_ids, attention_mask, tensor_label


class ELECTRAClassifier(nn.Module):
    def __init__(self,
                 electra,
                 hidden_size=768,
                 num_classes=2,
                 dr_rate=None,
                 params=None):

        super(ELECTRAClassifier, self).__init__()

        self.electra = electra

        # do not train electra parameters
        for p in self.electra.parameters():
            p.requires_grad = False

        self.dr_rate = dr_rate

        # self.classifier = nn.Linear(hidden_size , num_classes)

        #         # 방법 1 -> forward에서 처리해줘야 함.
        #         self.classifier1 = nn.Linear(hidden_size, 100) # y = Wx
        #         self.classifier2 = nn.Linear(100, num_classes) # z = Uy
        #         #layer 추가 시 activation function을 주지 않으면 의미가 없음.
        #         self.relu = nn.ReLU()

        # 방법 2 -> forward에서 별도 처리 필요 X
        self.classifier = nn.Sequential(nn.Linear(hidden_size, 100), nn.ReLU(), nn.Linear(100, num_classes))

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, input_ids, token_type_ids, attention_mask):
        # eval: drop out 중지, batch norm 고정과 같이 evaluation으로 모델 변경
        self.electra.eval()

        # gradient 계산을 중지
        with torch.no_grad():
            # ElectraModel은 pooled_output을 리턴하지 않는 것을 제외하고 BertModel과 유사합니다.
            #            x = self.electra(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask).pooler_output

            x = self.electra(input_ids=input_ids, token_type_ids=token_type_ids,
                             attention_mask=attention_mask).last_hidden_state[:, 0, :]
            # .last_hidden_state[:, 0, :]: [batch , CLS 위치, depth]

            # Sentence Embedding으로 무엇을 넣을까? CLS, average, ... (Bert의 경우에는 Sentence BERT라는 게 제안되었다고 함)

        x = self.dropout(x)

        return self.classifier(x)


def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

# KoELECTRA-Base
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
model = ElectraModel.from_pretrained("monologg/koelectra-base-discriminator")

df = pd.read_csv('data/twitter_clean.csv', encoding="UTF-8")
df['split'] = np.random.randn(df.shape[0], 1)
msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)

train = sentencEmojiDataset('data/train.csv', tokenizer)
test = sentencEmojiDataset('data/test.csv', tokenizer)

train_collate_fn = collate_fn(train.labels_dict)
test_collate_fn = collate_fn(test.labels_dict)

# Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-3

train_dataloader = DataLoader(train, batch_size=batch_size, collate_fn=train_collate_fn, shuffle = True, drop_last = True)
test_dataloader = DataLoader(test, batch_size=batch_size, collate_fn=test_collate_fn, shuffle = False, drop_last = False)

label = list(set(list(df.iloc[:,1])))

# model = BERTClassifier(model,  dr_rate=0.5, num_classes = len(label))
model = ELECTRAClassifier(model,  dr_rate=0.2, num_classes = len(label))

#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
# optimizer = AdamW(model.parameters(), lr=learning_rate)

#Class Imbalance 문제 해결을 위한 weighted cross entropy
class_weights = compute_class_weight(class_weight = 'balanced', classes = np.unique(train.labels), y = train.labels)
class_weights = torch.tensor(class_weights, dtype=torch.float)

loss_fn = nn.CrossEntropyLoss(weight = class_weights, reduction = 'mean')

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    loss_sum = 0
    model.train()
    for batch_id, (input_ids, token_type_ids, attention_mask, tensor_label) in enumerate(train_dataloader):
        optimizer.zero_grad()

        out = model(input_ids, token_type_ids, attention_mask)
        loss = loss_fn(out, tensor_label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        #         scheduler.step()  # Update learning rate schedule
        batch_acc = calc_accuracy(out, tensor_label)
        train_acc += batch_acc
        loss_sum += loss.data.cpu().numpy()

        print("epoch {} batch id {}/{} loss {} train acc {}".format(e + 1, batch_id + 1, len(train_dataloader),
                                                                    loss.data.cpu().numpy(), batch_acc))
    print("epoch {} train acc {} loss mean {}".format(e + 1, train_acc / (batch_id + 1),
                                                      loss_sum / len(train_dataloader)))
    model.eval()
    with torch.no_grad():
        for batch_id, (input_ids, token_type_ids, attention_mask, tensor_label) in enumerate(test_dataloader):

            out = model(input_ids, token_type_ids, attention_mask)
            test_acc += calc_accuracy(out, tensor_label)
    print("epoch {} test acc {}".format(e + 1, test_acc / (batch_id + 1)))

    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, 'pytorch_model.bin')

    # torch.save(model.state_dict(), 'pytorch_model.bin')
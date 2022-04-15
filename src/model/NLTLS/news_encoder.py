import torch
import torch.nn as nn
import numpy as np
from model.general.attention import AdditiveAttention


class NewsEncoder(torch.nn.Module):
    def __init__(self, args, word_embedding, docres):
        super(NewsEncoder, self).__init__()
        self.args = args
        self.docres = docres

        self.category_embedding = nn.Embedding(self.args.n_categories, self.args.category_embedding_dim, padding_idx=0)

        category_dense = [
            nn.Linear(self.args.category_embedding_dim, self.args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.category_dense = nn.Sequential(*category_dense)

        subcategory_dense = [
            nn.Linear(self.args.category_embedding_dim, self.args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.subcategory_dense = nn.Sequential(*subcategory_dense)

        docres_dense = [
            nn.Linear(50, self.args.n_filters),
            nn.ReLU(inplace=False)
        ]
        self.docres_dense = nn.Sequential(*docres_dense)

        if word_embedding is None:
            self.word_embedding = [
                nn.Embedding(self.args.n_words, self.args.word_embedding_dim, padding_idx=0),  # 若维度不够用0填充
                nn.Dropout(p=self.args.dropout, inplace=False)
            ]
        else:
            self.word_embedding = [
                nn.Embedding.from_pretrained(word_embedding, freeze=False, padding_idx=0),
                nn.Dropout(p=self.args.dropout, inplace=False)
            ]
        self.word_embedding = nn.Sequential(*self.word_embedding)

        title_CNN = [
            nn.Conv2d(1, self.args.n_filters, kernel_size=(self.args.window_size, self.args.word_embedding_dim),
                      padding=((self.args.window_size - 1) // 2, 0)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=self.args.dropout, inplace=False)
        ]
        self.title_CNN = nn.Sequential(*title_CNN)
        self.title_attention = AdditiveAttention(self.args.query_vector_dim, self.args.n_filters)

        abstract_CNN = [
            nn.Conv2d(1, self.args.n_filters, kernel_size=(self.args.window_size, self.args.word_embedding_dim),
                      padding=((args.window_size - 1) // 2, 0)),
            nn.ReLU(inplace=False),
            nn.Dropout(p=self.args.dropout, inplace=False)
        ]
        self.abstract_CNN = nn.Sequential(*abstract_CNN)
        self.abstract_attention = AdditiveAttention(args.query_vector_dim, self.args.n_filters)

        self.news_attn = AdditiveAttention(self.args.query_vector_dim, self.args.n_filters)

    def forward(self, news, doc):
        category = news['category'].to(self.args.device)
        category_embedding = self.category_embedding(category)
        category_dense = self.category_dense(category_embedding)

        subcategoty = news['subcategory'].to(self.args.device)
        subcategory_embedding = self.category_embedding(subcategoty)
        subcategory_dense = self.subcategory_dense(subcategory_embedding)

        topic_vector = self.docres[doc]
        topic_dense = self.docres_dense(topic_vector)

        title = torch.stack(news['title'], dim=1).to(self.args.device)
        title_mask_attn, title_mask_self_attn = self.masking(title, self.get_len(title))
        title_embedding = self.word_embedding(title)
        title_feature = self.title_CNN(title_embedding.unsqueeze(dim=1)).squeeze(dim=3)
        title_attn = self.title_attention(title_feature.transpose(1, 2), mask=title_mask_attn.to(self.args.device))

        abstract = torch.stack(news['abstract'], dim=1).to(self.args.device)
        abstract_mask_attn, abstract_mask_self_attn = self.masking(abstract, self.get_len(abstract))
        abstract_embedding = self.word_embedding(abstract)
        abstract_feature = self.abstract_CNN(abstract_embedding.unsqueeze(dim=1)).squeeze(dim=3)
        abstract_attn = self.abstract_attention(abstract_feature.transpose(1, 2), mask=abstract_mask_attn.to(self.args.device))

        news_vector = torch.stack([category_dense, subcategory_dense, topic_dense, title_attn, abstract_attn], dim=1)

        news_vector_attn = self.news_attn(news_vector)

        return news_vector_attn

    def masking(self, batch, batch_len):
        mask = np.zeros((batch.shape[0], batch.shape[1]), dtype=int)
        for i in range(len(mask)):
            mask[i][:batch_len[i]] = 1

        mask_attn = mask.copy()
        mask1 = mask[:, :, np.newaxis]
        mask2 = mask[:, np.newaxis, :]
        mask = np.matmul(mask1, mask2)

        return torch.IntTensor(mask_attn), torch.IntTensor(mask)

    def get_len(self, batch):
        length = []
        for i in batch:
            count = 0
            for j in reversed(i):
                if j != 0:
                    length.append(len(i) - count)
                    break
                else:
                    count += 1
                    if count == len(i):
                        length.append(0)

        return np.array(length)

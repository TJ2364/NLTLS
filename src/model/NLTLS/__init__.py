import torch
from model.NLTLS.news_encoder import NewsEncoder
from model.NLTLS.user_encoder import UserEncoder
from model.general.click_predictor import ClickPredictor
import numpy as np


class model(torch.nn.Module):
    def __init__(self, args, word_embedding, docres, doc_dict):
        super(model, self).__init__()
        self.args = args
        self.doc_dict = doc_dict

        self.news_encoder = NewsEncoder(args, word_embedding, docres)
        self.user_encoder = UserEncoder(args)
        self.click_predictor = ClickPredictor()

    def forward(self, batch):
        """NewsID for LDA"""
        his_id = np.array(batch['his_id'])
        his_new = []
        for row in his_id:
            his_li = [self.doc_dict[i] for i in row]
            his_new.append(his_li)
        his_new = np.array(his_new)
        his_id = torch.LongTensor(his_new).to(self.args.device)

        """User Representations"""
        browsed_vector = torch.stack([self.news_encoder(x, his_id[i]) for i, x in enumerate(batch['history'])], dim=1)
        batch_len = batch['seq_length']
        user_long_vector, user_short_vector = self.user_encoder(browsed_vector, batch_len)
        user_final_vector = torch.cat((user_long_vector, user_short_vector), dim=-1)

        """News Representation"""
        can_id = np.array(batch['candidate_id'])
        can_new = []
        for row in can_id:
            can_li = [self.doc_dict[i] for i in row]
            can_new.append(can_li)
        can_new = np.array(can_new)
        can_id = torch.LongTensor(can_new).to(self.args.device)
        candidate_vector = torch.stack([self.news_encoder(x, can_id[i]) for i, x in enumerate(batch['candidate'])], dim=1)

        """predictor"""
        batch_predict = self.click_predictor(candidate_vector, user_final_vector)

        return batch_predict

    def get_user_vector(self, his_vector, seq_len):
        user_long_vector, user_short_vector = self.user_encoder(his_vector, seq_len)
        user_final_vector = torch.cat((user_long_vector, user_short_vector), dim=-1)
        return user_final_vector

    def get_news_vector(self, candidate, can_id):
        can_id = np.array([self.doc_dict[i] for i in can_id])
        can_id = torch.LongTensor(can_id).to(self.args.device)
        news_vector = self.news_encoder(candidate, can_id)
        return news_vector

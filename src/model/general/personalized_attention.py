import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(torch.nn.Module):
    def __init__(self, query_vector_dim, input_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(input_vector_dim, query_vector_dim)

    def forward(self, input, query_vector):
        scores = torch.matmul(torch.tanh(self.linear(input)), query_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        weight = F.softmax(scores, dim=1)
        result = torch.bmm(weight.unsqueeze(dim=1), input).squeeze(dim=1)
        return result
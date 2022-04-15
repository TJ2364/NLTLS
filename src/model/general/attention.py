import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class AdditiveAttention(torch.nn.Module):
    def __init__(self, query_vector_dim, input_vector_dim):
        super(AdditiveAttention, self).__init__()
        self.linear = nn.Linear(input_vector_dim, query_vector_dim)
        self.query_vector = nn.Parameter(torch.empty(query_vector_dim).uniform_(-0.1, 0.1))

    def forward(self, input, mask=None):
        """
        Args:
            input: batch_size, n_input_vector, input_vector_dim
        Returns:
            result: batch_size, input_vector_dim
        """
        scores = torch.matmul(torch.tanh(self.linear(input)), self.query_vector)
        if mask is not None:
            scores = scores.masked_fill(Variable(mask) == 0, -1e9)
        weight = F.softmax(scores, dim=1)
        result = torch.bmm(weight.unsqueeze(dim=1), input).squeeze(dim=1)
        return result
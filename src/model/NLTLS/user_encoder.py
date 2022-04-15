import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from model.general.personalized_attention import AdditiveAttention


class UserEncoder(torch.nn.Module):
    def __init__(self, args):
        super(UserEncoder, self).__init__()
        self.args = args
        self.gru = nn.GRU(self.args.n_filters, self.args.n_filters//2, 1, bidirectional=False, batch_first=True)
        self.user_long_attn = AdditiveAttention(self.args.n_filters//2, self.args.n_filters//20)

    def forward(self, his_vector, his_len):
        his_len[his_len > self.args.n_browsed_histories] = self.args.n_browsed_histories
        his_len[his_len == 0] = 1

        rec_gru_input = pack_padded_sequence(his_vector, his_len, batch_first=True, enforce_sorted=False)

        out, hidden = self.gru(rec_gru_input)
        user_short_vector = hidden[-1]
        outputs, _ = pad_packed_sequence(out, batch_first=True)
        user_long_vector = self.user_long_attn(outputs, user_short_vector)

        return user_long_vector, user_short_vector

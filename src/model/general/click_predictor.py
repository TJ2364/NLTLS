import torch


class ClickPredictor(torch.nn.Module):
    def __init__(self):
        super(ClickPredictor, self).__init__()

    def forward(self, candidate_vector, user_vector):
        predict = torch.bmm(candidate_vector, user_vector.unsqueeze(dim=-1)).squeeze(dim=-1)
        return predict

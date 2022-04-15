import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cuda:0', help='cuda device')

parser.add_argument('--n_browsed_histories', type=int, default=50, help='number of browsed news per user')
parser.add_argument('-n_words', type=int, default=386002, help='number of words')  # MINDsmall
parser.add_argument('-n_categories', type=int, default=281, help='number of categories and subcategories')  # MINDsmall
parser.add_argument('--n_words_title', type=int, default=20, help='number of words per title')
parser.add_argument('--n_words_abstract', type=int, default=50, help='number of words per abstract')
parser.add_argument('--word_embedding_dim', type=int, default=300, help='dimension of word embedding vector')
parser.add_argument('-category_embedding_dim', type=int, default=100, help='dimension of category embedding vector')
parser.add_argument('--query_veeddctor_dim', type=int, default=200, help='dimension of the query vector in attention')

parser.add_argument('--batch_size', type=int, default=2, help='size of each batch')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--dropout', type=float, default=0.2, help='ratio of dropout')
parser.add_argument('--load', type=str, default=None, help='load pretrained model_7')

parser.add_argument('--n_filters', type=int, default=300, help='number of filters in CNN')
parser.add_argument('--window_size', type=int, default=3, help='size of filter in CNN')
parser.add_argument('--masking_probability', type=float, default=0.5, help='masking_probability')

# parser.add_argument('--n_users', type=int, default=94058, help='n_users')  # MINDsmall
# parser.add_argument('--n_news', type=int, default=65239, help='n_users')  # MINDsmall

args = parser.parse_args()

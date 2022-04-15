from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval
import torch


class TrainingDataset(Dataset):
    def __init__(self, args):
        super(TrainingDataset, self).__init__()
        self.args = args
        self.empty_news = {
            'category': 0,
            'subcategory': 0,
            'title': [0 for _ in range(self.args.n_words_title)],
            'abstract': [0 for _ in range(self.args.n_words_abstract)],
        }

        # 测试MINDsmall时需更改train_data.csv为train_small_data.csv
        self.behaviors = pd.read_table('../data/train_data.csv', na_filter=False)
        # 测试MINDsmall时需更改news.csv为news_small.csv
        self.news = pd.read_table('../data/news.csv', index_col='NewsID', na_filter=False)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, index):
        item = {}
        row = self.behaviors.iloc[index]
        if row.Histories == ' ':
            tmp = []
            length = 0
        else:
            tmp = row.Histories.split(' ')
            length = len(tmp)

        if length < self.args.n_browsed_histories:
            repeat_times = self.args.n_browsed_histories - length
            item['history'] = [self.get_news_dict(news_id) for news_id in tmp]
            item['his_id'] = tmp
            item['history'] += [self.empty_news for _ in range(repeat_times)]
            item['his_id'] += ['0' for _ in range(repeat_times)]
        else:
            item['history'] = [self.get_news_dict(news_id) for news_id in tmp[-self.args.n_browsed_histories:]]
            item['his_id'] = tmp[-self.args.n_browsed_histories:]

        tmp = row.Impressions.split(' ')
        item['candidate_id'] = tmp
        item['candidate'] = [self.get_news_dict(NewsID) for NewsID in tmp]

        item['user_id'] = row.UserID

        item['seq_length'] = length

        return item

    def get_news_dict(self, NewsID):
        if NewsID in self.news.index:
            row = self.news.loc[NewsID]
            news = {
                'category': row.Category,
                'subcategory': row.SubCategory,
                'title': literal_eval(row.Title),
                'abstract': literal_eval(row.Abstract)
            }

        else:
            news = self.empty_news

        return news


class TestUserDataset(Dataset):
    def __init__(self, args, total_test_news, total_news_id, typename):
        super(TestUserDataset, self).__init__()
        self.args = args
        self.typename = typename
        self.total_test_news = total_test_news
        self.total_news_id = total_news_id

        if self.typename == 'valid':
            # 测试MINDsmall时需更改valid_data.csv为valid_small_data.csv
            self.behaviors = pd.read_table('../data/valid_data.csv', na_filter=False)
        elif self.typename == 'test':
            # 测试MINDsmall时需更改test_data.csv为test_small_data.csv
            self.behaviors = pd.read_table('../data/test_data.csv', na_filter=False)
        else:
            print('typename error!')

        self.empty_news_vector = self.total_test_news[self.total_news_id['0']].unsqueeze(dim=0)

    def __len__(self):
        return len(self.behaviors)

    def __getitem__(self, index):
        item = {}
        row = self.behaviors.iloc[index]
        if row.Histories == ' ':
            length = 0
            item['history'] = torch.cat([self.empty_news_vector for _ in range(self.args.n_browsed_histories)])
            item['his_id'] = ['0' for _ in range(self.args.n_browsed_histories)]
        else:
            tmp = row.Histories.split(' ')
            length = len(tmp)
            if length < self.args.n_browsed_histories:
                repeat_times = self.args.n_browsed_histories - length
                item['history'] = torch.cat([self.total_test_news[self.total_news_id.setdefault(i, 0)].unsqueeze(dim=0)
                                             for i in tmp])
                item['his_id'] = tmp
                item['history'] = torch.cat([item['history'], torch.cat([self.empty_news_vector for _ in range(repeat_times)])])
                item['his_id'] += ['0' for _ in range(repeat_times)]
            else:
                item['history'] = torch.cat([self.total_test_news[self.total_news_id.setdefault(i, 0)].unsqueeze(dim=0)
                                             for i in tmp[-self.args.n_browsed_histories:]])
                item['his_id'] = tmp[-self.args.n_browsed_histories:]

        item['user_id'] = row.UserID
        item['seq_length'] = length

        return item


class TestNewsDataset(Dataset):
    def __init__(self):
        super(TestNewsDataset, self).__init__()
        # 测试MINDsmall时需更改news.csv为news_small.csv
        self.news = pd.read_table('../data/news.csv', na_filter=False)

    def __len__(self):
        return len(self.news)

    def __getitem__(self, index):
        row = self.news.iloc[index]
        item = {
            'category': row.Category,
            'subcategory': row.SubCategory,
            'title': literal_eval(row.Title),
            'abstract': literal_eval(row.Abstract)
        }

        return row.NewsID, item




































import pandas as pd
import numpy as np
import random
import datetime


def preprocess_train(data):
    mat_data = []

    group_user = data.groupby('UserID')
    for key in group_user.groups.keys():
        user_list = []
        user_list.append(key)

        user_data = group_user.get_group(key)

        user_long_his = user_data['Histories'].iloc[0]
        user_list.append(user_long_his)

        user_pre = user_data['Impressions'].str.split(' ')
        for i in user_pre:
            pre_true = []
            pre_false = []
            flag = 0
            for j in i:
                if j.endswith('-1'):
                    flag += 1
                    pre_true.append(j.split('-')[0])
                else:
                    pre_false.append(j.split('-')[0])

            if flag != 0:
                if len(pre_false) >= 4:
                    for i in pre_true:
                        li = [i]
                        pre_false_sample = random.sample(pre_false, 4)
                        li.extend(pre_false_sample)
                        li = ' '.join(li)
                        user_li = user_list.copy()
                        user_li.append(li)
                        mat_data.append(user_li)
                else:
                    for i in pre_true:
                        li = [i]
                        pre_false_sample = pre_false + list(
                            np.random.choice(pre_false, 4 - len(pre_false)))
                        li.extend(pre_false_sample)
                        li = ' '.join(li)
                        user_li = user_list.copy()
                        user_li.append(li)
                        mat_data.append(user_li)

    random.shuffle(mat_data)
    mat_data = pd.DataFrame(columns=['UserID', 'Histories', 'Impressions'], data=mat_data)
    mat_data.to_csv('../data/train_data.csv', sep='\t', index=False)


def train_valid_split(data):
    meta_data_train = []
    meta_data_valid = []

    group_user = data.groupby('UserID')
    for key in group_user.groups.keys():
        user_list = []
        user_list.append(key)
        user_data = group_user.get_group(key)

        user_long_his = user_data['Histories'].iloc[0]
        user_list.append(user_long_his)

        user_data = user_data.sort_values(['Time'], ascending=True)
        user_data['date'] = user_data['Time'].apply(lambda x: x.date())
        valid_date = datetime.date(2019, 11, 14)

        user_valid = user_data.loc[user_data['date'] == valid_date]
        for index, i in user_valid.iterrows():
            user_pre_each = i['Impressions'].split(' ')
            pre = []
            pre_label = []
            for j in user_pre_each:
                pre.append(j.split('-')[0])
                pre_label.append(j.split('-')[1])
            pre = ' '.join(pre)
            pre_label = ' '.join(pre_label)
            user_li = user_list.copy()
            user_li.append(pre)
            user_li.append(pre_label)
            user_li.append(i['ImpressionID'])
            meta_data_valid.append(user_li)

        user_train = user_data.loc[user_data['date'] != valid_date]
        user_pre = user_train['Impressions'].str.split(' ')
        for i in user_pre:
            pre_true = []
            pre_false = []
            flag = 0
            for j in i:
                if j.endswith('-1'):
                    flag += 1
                    pre_true.append(j.split('-')[0])
                else:
                    pre_false.append(j.split('-')[0])

            if flag != 0:
                if len(pre_false) >= 4:
                    for i in pre_true:
                        li = [i]
                        pre_false_sample = random.sample(pre_false, 4)
                        li.extend(pre_false_sample)
                        li = ' '.join(li)
                        user_li = user_list.copy()
                        user_li.append(li)
                        meta_data_train.append(user_li)
                else:
                    for i in pre_true:
                        li = [i]
                        pre_false_sample = pre_false + list(
                            np.random.choice(pre_false, 4 - len(pre_false)))
                        li.extend(pre_false_sample)
                        li = ' '.join(li)
                        user_li = user_list.copy()
                        user_li.append(li)
                        meta_data_train.append(user_li)

    random.shuffle(meta_data_train)
    meta_data_train = pd.DataFrame(columns=['UserID', 'Histories', 'Impressions'], data=meta_data_train)
    meta_data_train.to_csv('../data/train_small_data.csv', sep='\t', index=False)

    meta_data_valid = pd.DataFrame(columns=['UserID', 'Histories', 'Impressions', 'Label', 'ImpressionID'], data=meta_data_valid)
    meta_data_valid.to_csv('../data/valid_small_data.csv', sep='\t', index=False)


def preprocess_valid(data, data_name):
    mat_data = []

    group_user = data.groupby('UserID')

    for key in group_user.groups.keys():
        user_list = []
        user_list.append(key)

        user_data = group_user.get_group(key)

        user_long_his = user_data['Histories'].iloc[0]
        user_list.append(user_long_his)

        for index, i in user_data.iterrows():
            user_pre_each = i['Impressions'].split(' ')
            pre = []
            pre_label = []

            for j in user_pre_each:
                pre.append(j.split('-')[0])
                pre_label.append(j.split('-')[1])
            pre = ' '.join(pre)
            pre_label = ' '.join(pre_label)
            user_li = user_list.copy()
            user_li.append(pre)
            user_li.append(pre_label)
            user_li.append(i['ImpressionID'])
            mat_data.append(user_li)

    mat_data = pd.DataFrame(columns=['UserID', 'Histories', 'Impressions', 'Label', 'ImpressionID'], data=mat_data)
    mat_data.to_csv('../data/' + data_name + '_data.csv', sep='\t', index=False)


if __name__ == '__main__':
    """MIND"""
    behaviors_train = pd.read_csv('../dataset/MINDlarge_train/behaviors.tsv', sep='\t', header=None)
    behaviors_train.columns = ['ImpressionID', 'UserID', 'Time', 'Histories', 'Impressions']
    behaviors_train.Histories.fillna(' ', inplace=True)
    preprocess_train(behaviors_train)

    behaviors_dev = pd.read_csv('../dataset/MINDlarge_dev/behaviors.tsv', sep='\t', header=None)
    behaviors_dev.columns = ['ImpressionID', 'UserID', 'Time', 'Histories', 'Impressions']
    behaviors_dev.Histories.fillna(' ', inplace=True)
    preprocess_valid(behaviors_dev, 'valid')

    behaviors_test = pd.read_csv('../dataset/MINDlarge_test/behaviors.tsv', sep='\t', header=None)
    behaviors_test.columns = ['ImpressionID', 'UserID', 'Time', 'Histories', 'Impressions']
    behaviors_test.Histories.fillna(' ', inplace=True)
    behaviors_test.to_csv('../data/test_data.csv', sep='\t', index=False,
                          columns=['UserID', 'Histories', 'Impressions', 'ImpressionID'])

    """MINDsmall"""
    behaviors_train = pd.read_csv('../dataset/MINDsmall_train/behaviors.tsv', sep='\t', header=None)
    behaviors_train.columns = ['ImpressionID', 'UserID', 'Time', 'Histories', 'Impressions']
    behaviors_train['Time'] = pd.to_datetime(behaviors_train['Time'], format='%m/%d/%Y %I:%M:%S %p')
    behaviors_train.Histories.fillna(' ', inplace=True)
    train_valid_split(behaviors_train)

    behaviors_small_dev = pd.read_csv('../dataset/MINDsmall_dev/behaviors.tsv', sep='\t', header=None)
    behaviors_small_dev.columns = ['ImpressionID', 'UserID', 'Time', 'Histories', 'Impressions']
    behaviors_small_dev.Histories.fillna(' ', inplace=True)
    preprocess_valid(behaviors_small_dev, 'test_small')












import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import numpy as np
from src.args import args
from nltk.corpus import stopwords
from string import punctuation


def process_news(data, data_name=None):
    news_docs = pd.read_table('../dataset/news.csv', header=None, names=['body'], na_filter=False)
    for row in news_docs.itertuples():
        if row.Index == 0:
            continue
        tmp = row.body
        body = tmp[:-7]
        nid = tmp[-7:]
        news_docs.at[row.Index, 'body'] = body
        news_docs.at[row.Index, 'nid'] = nid
    news_docs = news_docs.drop(labels=0)
    news_docs = news_docs.drop_duplicates().reset_index(drop=True)
    news_docs.set_index(['nid'], inplace=True)

    body = []
    with tqdm(total=len(data), desc='Processing body') as p:
        for row in data.itertuples():
            nid = row.Body.split('/')[-1].split('.')[0]
            if nid not in news_docs.index:
                body.append(' ')
                continue
            news_body = news_docs.loc[nid, 'body']
            body.append(news_body)
            p.update(1)

    data['Body'] = body

    body_list = []
    category_dict, word_dict = {}, {}
    stop_words = stopwords.words('english')
    punctuations = list(punctuation)
    stop_words.extend(punctuations)

    with tqdm(total=len(data), desc='Processing News') as p:
        for row in data.itertuples():
            if row.Category not in category_dict:
                category_dict[row.Category] = len(category_dict) + 1
            if row.SubCategory not in category_dict:
                category_dict[row.SubCategory] = len(category_dict) + 1

            for word in word_tokenize(row.Title.lower()):
                if word not in stop_words and word not in word_dict:
                    word_dict[word] = len(word_dict) + 1
            for word in word_tokenize(row.Abstract.lower()):
                if word not in stop_words and word not in word_dict:
                    word_dict[word] = len(word_dict) + 1

            body_li = []
            for word in word_tokenize(row.Body.lower()):
                if word not in stop_words and word not in word_dict:
                    word_dict[word] = len(word_dict) + 1
                if word not in stop_words:
                    body_li.append(word)
            body_list.append(body_li)

            p.update(1)

    data['Body'] = body_list

    if data_name == 'small':
        pd.DataFrame(category_dict.items(), columns=['category', 'index']).to_csv('../data/category_small_dict.csv',
                                                                                  sep='\t',
                                                                                  index=False)
        pd.DataFrame(word_dict.items(), columns=['word', 'index']).to_csv('../data/word_small_dict.csv', sep='\t',
                                                                          index=False)
        data.to_csv('../data/news_small_lda.csv', sep='\t', index=False, columns=['NewsID', 'Body'])
    else:
        pd.DataFrame(category_dict.items(), columns=['category', 'index']).to_csv('../data/category_dict.csv', sep='\t',
                                                                                  index=False)
        pd.DataFrame(word_dict.items(), columns=['word', 'index']).to_csv('../data/word_dict.csv', sep='\t',
                                                                          index=False)

        data.to_csv('../data/news_lda.csv', sep='\t', index=False, columns=['NewsID', 'Body'])

    print('\ttotal_category:', len(category_dict))
    print('\ttotal_word:', len(word_dict))
    print('\tRemember to set n_categories = {} and n_words = {} in option.py'.format(len(category_dict) + 1,
                                                                                     len(word_dict) + 1))

    with tqdm(total=len(data), desc='Encoding news') as p:
        for row in data.itertuples():
            data.at[row.Index, 'Category'] = category_dict[row.Category]
            data.at[row.Index, 'SubCategory'] = category_dict[row.SubCategory]

            title = []
            abstract = []
            try:
                for word in word_tokenize(row.Title.lower()):
                    if word in word_dict:
                        title.append(word_dict[word])
                if len(title) < args.n_words_title:
                    repeat_times = args.n_words_title - len(title)
                    title += [0 for _ in range(repeat_times)]
                else:
                    title = title[:args.n_words_title]
            except IndexError:
                pass
            try:
                for word in word_tokenize(row.Abstract.lower()):
                    if word in word_dict:
                        abstract.append(word_dict[word])
                if len(abstract) < args.n_words_abstract:
                    repeat_times = args.n_words_abstract - len(abstract)
                    abstract += [0 for _ in range(repeat_times)]
                else:
                    abstract = abstract[:args.n_words_abstract]
            except IndexError:
                pass

            data.at[row.Index, 'Title'] = title
            data.at[row.Index, 'Abstract'] = abstract

            p.update(1)

    data = data.append(
        [{'NewsID': 0, 'Category': 0, 'SubCategory': 0, 'Title': [0 for _ in range(args.n_words_title)],
          'Abstract': [0 for _ in range(args.n_words_abstract)]}], ignore_index=True)

    if data_name == 'small':
        data.to_csv('../data/news_small.csv', sep='\t', index=False,
                    columns=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract'])
    else:
        data.to_csv('../data/news.csv', sep='\t', index=False,
                    columns=['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract'])

    NewsID = data['NewsID']
    news_dict = dict(zip(NewsID, range(len(NewsID))))
    if data_name == 'small':
        pd.DataFrame(news_dict.items(), columns=['news', 'index']).to_csv('../data/news_small_dict.csv', sep='\t', index=False)
    else:
        pd.DataFrame(news_dict.items(), columns=['news', 'index']).to_csv('../data/news_dict.csv', sep='\t', index=False)

    print('Finish news preprocessing')


def word_embedding(data_name=None):
    print('Start word embedding...')

    if data_name == 'small':
        word_dict = dict(
            pd.read_csv('../data/word_small_dict.csv', sep='\t', na_filter=False, header=0).values.tolist())
    else:
        word_dict = dict(
            pd.read_csv('../data/word_dict.csv', sep='\t', na_filter=False, header=0).values.tolist())

    glove_embedding = pd.read_table('../dataset/glove.840B.300d.txt', sep=' ', header=None, index_col=0, quoting=3)

    embedding_result = np.random.normal(size=(len(word_dict) + 1, 300))
    word_missing = 0
    with tqdm(total=len(word_dict), desc="Generating word embedding") as p:
        for k, v in word_dict.items():
            if k in glove_embedding.index:
                embedding_result[v] = glove_embedding.loc[k].tolist()
            else:
                word_missing += 1
            p.update(1)

    if data_name == 'small':
        np.save('../data/word_small_embedding.npy', embedding_result)
    else:
        np.save('../data/word_embedding.npy', embedding_result)

    print('\ttotal_missing_word:', word_missing)
    print('Finish word embedding')


if __name__ == '__main__':
    """MIND"""
    # news_train = pd.read_csv('../dataset/MINDlarge_train/news.tsv', sep='\t', header=None)
    # news_train.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'Body', 'TitleEntities',
    #                       'AbstractEntities']
    #
    # news_dev = pd.read_csv('../dataset/MINDlarge_dev/news.tsv', sep='\t', header=None)
    # news_dev.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'Body', 'TitleEntities',
    #                     'AbstractEntities']
    #
    # news_test = pd.read_csv('../dataset/MINDlarge_test/news.tsv', sep='\t', header=None)
    # news_test.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'Body', 'TitleEntities',
    #                      'AbstractEntities']
    #
    # mind_data = pd.concat((news_train, news_dev, news_test), axis=0)
    # mind_data = mind_data.drop_duplicates()
    # mind_data = mind_data.reset_index(drop=True)
    # mind_data.Title.fillna(' ', inplace=True)
    # mind_data.Abstract.fillna(' ', inplace=True)
    # mind_data.Body.fillna(' ', inplace=True)
    # process_news(mind_data)
    # word_embedding()

    """MINDsmall"""
    news_small_train = pd.read_csv('../dataset/MINDsmall_train/news.tsv', sep='\t', header=None)
    news_small_train.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'Body', 'TitleEntities',
                                'AbstractEntities']

    news_small_dev = pd.read_csv('../dataset/MINDsmall_dev/news.tsv', sep='\t', header=None)
    news_small_dev.columns = ['NewsID', 'Category', 'SubCategory', 'Title', 'Abstract', 'Body', 'TitleEntities',
                              'AbstractEntities']

    mind_small_data = pd.concat((news_small_train, news_small_dev), axis=0)
    mind_small_data = mind_small_data.drop_duplicates()
    mind_small_data = mind_small_data.reset_index(drop=True)
    mind_small_data.Title.fillna(' ', inplace=True)
    mind_small_data.Abstract.fillna(' ', inplace=True)
    mind_small_data.Body.fillna(' ', inplace=True)
    process_news(mind_small_data, data_name='small')
    word_embedding(data_name='small')

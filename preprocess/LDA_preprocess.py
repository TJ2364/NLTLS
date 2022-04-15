import pickle
from time import time
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


if __name__ == '__main__':
    """MIND"""
    news_lda = pd.read_table('../data/news_lda.csv', na_filter=False)  # 测试MINDsmall时需更改news_lda.csv为news_small_lda.csv
    body = np.array(news_lda['Body'])
    body = [literal_eval(i) for i in body]
    body.append([])
    body = [' '.join(i) for i in body]

    corpus = body
    cntVector = CountVectorizer()
    cntTf = cntVector.fit_transform(corpus)

    lda = LatentDirichletAllocation(n_components=50, max_iter=100, learning_method='batch')
    t0 = time()
    docres = lda.fit_transform(cntTf)
    print("done in %0.3fs" % (time() - t0))
    # save model
    f = open('../data/docres.pickle', 'wb')
    pickle.dump(docres, f)
    f.close()


## NLTLS
#### 1. 下载数据集MIND、MINDsmall，保存到dataset目录（网址：https://msnews.github.io/）
#### 2. 下载glove.840B.300d，保存到dataset目录（网址：https://nlp.stanford.edu/projects/glove/）
#### 3. 爬取新闻正文，命名为news.csv，保存到dataset目录（网址：https://github.com/msnews/MIND/tree/master/crawler）
MIND数据集不提供body，只给出了相应网址，可利用微软新闻所发布的官方API爬取相应正文文本，想正确爬取正文需按要求更改代码
#### 4. 运行preprocess目录中的数据预处理代码
#### 5. 运行src目录下的main.py
注意当下要测试的是MIND还是MINDsmall，需按提示更改代码
#### 6. MIND数据集不提供测试集标签，需到排行榜中提交预测结果并获得分数（网址：https://competitions.codalab.org/competitions/24122#results）
需注册账号，并按正确格式提交预测文件

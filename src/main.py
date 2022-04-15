import pickle
from train_validation import *
from model.NLTLS import *
import numpy as np
import torch.nn as nn


if __name__ == '__main__':
    """MIND"""
    try:
        # 测试MINDsmall时需更改word_embedding.npy为word_small_embedding.npy
        word_embedding = torch.from_numpy(np.load('../data/word_embedding.npy')).float()
    except FileNotFoundError:
        word_embedding = None
        print('word_embedding is None')

    train_loader = DataLoader(TrainingDataset(args), batch_size=args.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    news_loader = DataLoader(TestNewsDataset(), batch_size=args.batch_size, shuffle=False, num_workers=0,
                              pin_memory=False, drop_last=False)

    """LDA"""
    f = open('../data/docres.pickle', 'rb')
    docres = pickle.load(f)
    f.close()
    docres = torch.tensor(docres, dtype=torch.float32).to(args.device)

    # 测试MINDsmall时需更改news_dict.csv为news_small_dict.csv
    doc_dic = pd.read_table('../data/news_dict.csv', na_filter=False)
    doc_id = doc_dic['news']
    index = doc_dic['index']
    doc_dict = dict(zip(doc_id, index))

    model = model(args, word_embedding, docres, doc_dict).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=args.betas, eps=args.epsilon)
    criterion = nn.CrossEntropyLoss()
    epoch = 0
    training_loss_epoch = []
    while True:
        epoch += 1
        training_loss = train(model, optimizer, criterion, train_loader)
        training_loss_epoch.append(training_loss)
        print('The average loss of training set for the first ' + str(epoch) + ' epochs: ' + str(training_loss_epoch))
        torch.save(model, '../result/model_NLTLS_train' + str(epoch) + '.pkl')

        evaluate(model, news_loader, 'valid')
        torch.save(model, '../result/model_NLTLS_valid' + str(epoch) + '.pkl')

        evaluate(model, news_loader, 'test')

        if epoch == 1:
            break



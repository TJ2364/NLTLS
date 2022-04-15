from args import args
from torch.utils.data import DataLoader
from metrics import *
from tqdm import tqdm
from dataset import *


def train(model, optimizer, criterion, train_loader):
    model.train()
    loss_list = []

    with tqdm(total=len(train_loader), desc='Training') as p:
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_predict = model(batch)
            batch_label = torch.zeros(len(batch_predict)).long().to(args.device)
            loss = criterion(batch_predict, batch_label)
            loss_list.append(loss.item())
            if i % 100 == 0 or i + 1 == len(train_loader):
                print('Loss: ' + str(np.mean(loss_list)))
            loss.backward()
            optimizer.step()

            p.update(1)

    average_loss = np.mean(loss_list)
    return average_loss


def evaluate(model, valid_loader, typename):
    with torch.no_grad():
        model.eval()
        AUC_list, MRR_list, nDCG5_list, nDCG10_list = [], [], [], []
        total_news_id = {}
        ranking = []
        impressionIDs = []
        with tqdm(total=len(valid_loader), desc='Generating total valid news') as p:
            for i, batch in enumerate(valid_loader):
                news_id_batch, news_batch = batch
                news_vector = model.get_news_vector(news_batch, news_id_batch)
                if i == 0:
                    total_test_news = news_vector
                else:
                    total_test_news = torch.cat([total_test_news, news_vector])
                for news_id in news_id_batch:
                    total_news_id[news_id] = len(total_news_id)

                p.update(1)

        test_user_loader = DataLoader(TestUserDataset(args, total_test_news, total_news_id, typename),
                                      batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False,
                                      drop_last=False)

        with tqdm(total=len(test_user_loader), desc='Generating total valid users') as p:
            for i, batch in enumerate(test_user_loader):
                his_vector = batch['history']
                seq_len = batch['seq_length']
                user_vector = model.get_user_vector(his_vector, seq_len)
                if i == 0:
                    total_test_user = user_vector
                else:
                    total_test_user = torch.cat([total_test_user, user_vector])

                p.update(1)

        if typename == 'valid':
            # 测试MINDsmall时需更改valid_data.csv为valid_small_data.csv
            behaviors = pd.read_table('../data/valid_data.csv', na_filter=False,
                                      usecols=[2, 3, 4], names=['impressions', 'y_true', 'ImpressionID'], header=0)
        elif typename == 'test':
            # 测试MINDsmall时需更改test_data.csv为test_small_data.csv
            behaviors = pd.read_table('../data/test_data.csv', na_filter=False,
                                      usecols=[2, 3, 4], names=['impressions', 'y_true', 'ImpressionID'], header=0)

        with tqdm(total=len(behaviors), desc='Validating') as p:
            for row in behaviors.itertuples():
                user_vector = total_test_user[row.Index]
                tmp = row.impressions.split(' ')
                news_vector = torch.cat([total_test_news[total_news_id[i]].unsqueeze(dim=0) for i in tmp])

                # 测试MINDsmall时只走valid分支，可屏蔽掉test分支
                if typename == 'valid':
                    y_true = [int(x) for x in row.y_true.split(' ')]
                    predict = torch.matmul(news_vector, user_vector).tolist()
                    AUC_list.append(AUC(y_true, predict))
                    MRR_list.append(MRR(y_true, predict))
                    nDCG5_list.append(nDCG(y_true, predict, 5))
                    nDCG10_list.append(nDCG(y_true, predict, 10))
                elif typename == 'test':
                    imp_id = row.ImpressionID
                    predict = torch.matmul(news_vector, user_vector).tolist()
                    predict = np.array(predict)
                    rank = predict.argsort().argsort() + 1
                    rank = list(len(rank) + 1 - rank)
                    ranking.append(rank)
                    impressionIDs.append(imp_id)

                p.update(1)

        # 测试MINDsmall时只走valid分支，可屏蔽掉test分支
        if typename == 'valid':
            print('AUC:', np.mean(AUC_list))
            print('MRR:', np.mean(MRR_list))
            print('nDCG@5:', np.mean(nDCG5_list))
            print('nDCG@10:', np.mean(nDCG10_list))
        elif typename == 'test':
            df = pd.DataFrame(list(zip(impressionIDs, ranking)), columns=['ImpressionID', 'Rank'])
            df = df.sort_values(by=['ImpressionID'], ascending=True)
            f = open('../result/prediction.txt', "a+")
            for i, j in df.iterrows():
                f.writelines((str(j['ImpressionID']), ' ', str(j['Rank']).replace(' ', ''), '\n'))
            print('Ranking produced.')
            f.close()

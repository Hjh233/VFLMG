import argparse
import time

import numpy as np
import pandas as pd

import mixed
from preprocess import setup_seed, LoadData, PrepareSimpleVFLDataset, get_dataloader


def joint_mutual_information(dataset, vfl_num_feature, n_splits, participants:list):
    '''
        e.g. vfl_num_feature = 80, n_splits = 8, participants = [2,5]
        then joint_data's feature = [10:20 \cup 40:50]
    '''
    x_train, y_train = LoadData('{}_train'.format(dataset))

    train_dataset = PrepareSimpleVFLDataset(x_train, y_train, vfl_num_feature)
    selected_x = train_dataset.input_x
    selected_y = train_dataset.y

    step = vfl_num_feature // n_splits

    count = 0

    for participant in participants:
        begin_idx = (participant - 1) * step
        if participant != n_splits:
            end_idx = participant * step
        else:
            end_idx = vfl_num_feature

        if count == 0:
            joint_data = selected_x[:, begin_idx:end_idx]
            count += 1
        else:
            joint_data = np.concatenate((joint_data, selected_x[:, begin_idx:end_idx]), 1)
                
    joint_mutual_information = mixed.Mixed_KSG(joint_data, selected_y)

    return joint_mutual_information



participants8 = [[1,2,3,4,5,6,7,8]]

participants5 = [[1, 2, 3, 4, 5], [1, 2, 3, 4, 6], [1, 2, 3, 4, 7], [1, 2, 3, 4, 8], 
                 [1, 2, 3, 5, 6], [1, 2, 3, 5, 7], [1, 2, 3, 5, 8], 
                 [1, 2, 3, 6, 7], [1, 2, 3, 6, 8], 
                 [1, 2, 3, 7, 8], 
                 [1, 2, 4, 5, 6], [1, 2, 4, 5, 7], [1, 2, 4, 5, 8], 
                 [1, 2, 4, 6, 7], [1, 2, 4, 6, 8], 
                 [1, 2, 4, 7, 8], 
                 [1, 2, 5, 6, 7], [1, 2, 5, 6, 8], 
                 [1, 2, 5, 7, 8], 
                 [1, 2, 6, 7, 8], 
                 [1, 3, 4, 5, 6], [1, 3, 4, 5, 7], [1, 3, 4, 5, 8], 
                 [1, 3, 4, 6, 7], [1, 3, 4, 6, 8], 
                 [1, 3, 4, 7, 8], 
                 [1, 3, 5, 6, 7], [1, 3, 5, 6, 8], 
                 [1, 3, 5, 7, 8], 
                 [1, 3, 6, 7, 8], 
                 [1, 4, 5, 6, 7], [1, 4, 5, 6, 8], 
                 [1, 4, 5, 7, 8], 
                 [1, 4, 6, 7, 8], 
                 [1, 5, 6, 7, 8], 
                 [2, 3, 4, 5, 6], [2, 3, 4, 5, 7], [2, 3, 4, 5, 8], 
                 [2, 3, 4, 6, 7], [2, 3, 4, 6, 8], 
                 [2, 3, 4, 7, 8], 
                 [2, 3, 5, 6, 7], [2, 3, 5, 6, 8], 
                 [2, 3, 5, 7, 8], 
                 [2, 3, 6, 7, 8], 
                 [2, 4, 5, 6, 7], [2, 4, 5, 6, 8], 
                 [2, 4, 5, 7, 8], 
                 [2, 4, 6, 7, 8], 
                 [2, 5, 6, 7, 8], 
                 [3, 4, 5, 6, 7], [3, 4, 5, 6, 8], 
                 [3, 4, 5, 7, 8], 
                 [3, 4, 6, 7, 8], 
                 [3, 5, 6, 7, 8], 
                 [4, 5, 6, 7, 8]]


participants4 = [[1,2,3,4],[1,2,3,5],[1,2,3,6],[1,2,3,7],[1,2,3,8],
                    [1,2,4,5],[1,2,4,6],[1,2,4,7],[1,2,4,8],
                    [1,2,5,6],[1,2,5,7],[1,2,5,8],
                    [1,2,6,7],[1,2,6,8],
                    [1,2,7,8],
                    [1,3,4,5],[1,3,4,6],[1,3,4,7],[1,3,4,8],
                    [1,3,5,6],[1,3,5,7],[1,3,5,8],
                    [1,3,6,7],[1,3,6,8],
                    [1,3,7,8],
                    [1,4,5,6],[1,4,5,7],[1,4,5,8],
                    [1,4,6,7],[1,4,6,8],
                    [1,4,7,8],
                    [1,5,6,7],[1,5,6,8],
                    [1,5,7,8],
                    [1,6,7,8],
                    [2,3,4,5],[2,3,4,6],[2,3,4,7],[2,3,4,8],
                    [2,3,5,6],[2,3,5,7],[2,3,5,8],
                    [2,3,6,7],[2,3,6,8],
                    [2,3,7,8],
                    [2,4,5,6],[2,4,5,7],[2,4,5,8],
                    [2,4,6,7],[2,4,6,8],
                    [2,4,7,8],
                    [2,5,6,7],[2,5,6,8],
                    [2,5,7,8],
                    [2,6,7,8],
                    [3,4,5,6],[3,4,5,7],[3,4,5,8],
                    [3,4,6,7],[3,4,6,8],
                    [3,4,7,8],
                    [3,5,6,7],[3,5,6,8],
                    [3,4,7,8],
                    [3,6,7,8],
                    [4,5,6,7],[4,5,6,8],
                    [4,5,7,8],
                    [4,6,7,8],
                    [5,6,7,8]]


participants3 = [[1,2,3],[1,2,4],[1,2,5],[1,2,6],[1,2,7],[1,2,8],
                 [1,3,4],[1,3,5],[1,3,6],[1,3,7],[1,3,8],
                 [1,4,5],[1,4,6],[1,4,7],[1,4,8],
                 [1,5,6],[1,5,7],[1,5,8],
                 [1,6,7],[1,6,8],
                 [1,7,8],
                 [2,3,4],[2,3,5],[2,3,6],[2,3,7],[2,3,8],
                 [2,4,5],[2,4,6],[2,4,7],[2,4,8],
                 [2,5,6],[2,5,7],[2,5,8],
                 [2,6,7],[2,6,8],
                 [2,7,8],
                 [3,4,5],[3,4,6],[3,4,7],[3,4,8],
                 [3,5,6],[3,5,7],[3,5,8],
                 [3,6,7],[3,6,8],
                 [3,7,8],
                 [4,5,6],[4,5,7],[4,5,8],
                 [4,6,7],[4,6,8],
                 [4,7,8],
                 [5,6,7],[5,6,8],
                 [5,7,8],
                 [6,7,8]]
    
participants2 = [[1,2],[1,3],[1,4],[1,5],[1,6],[1,7],[1,8],
                 [2,3],[2,4],[2,5],[2,6],[2,7],[2,8],
                 [3,4],[3,5],[3,6],[3,7],[3,8],
                 [4,5],[4,6],[4,7],[4,8],
                 [5,6],[5,7],[5,8],
                 [6,7],[6,8],
                 [7,8]]
    
participants1 = [[1], [2], [3], [4], [5], [6], [7], [8]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--vfl_num_feature', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--n_splits', type=int, required=True)

    args = parser.parse_args()

    dataset = args.dataset
    vfl_num_feature = args.vfl_num_feature
    n_splits = args.n_splits
    random_seed = args.seed

    setup_seed(random_seed)

    random_participants = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [1, 2, 3, 4, 5, 6, 7, 8, 9],
                           [1, 2, 3, 4, 6, 8, 9], [1, 2, 4, 5, 6], [1, 3, 6, 7], [2], [2, 4, 6, 7, 9], 
                           [1, 6], [1, 2, 3, 5, 6, 7, 8], [8, 3], [1, 2, 3, 6, 7], [2, 3, 4, 6, 7, 9], 
                           [1, 3, 5, 7, 9], [8, 5, 9], [1, 2, 3, 5, 9], [1, 2, 3, 5, 6, 9], [2, 3, 4, 8, 9], 
                           [1, 2], [2, 5, 7, 9], [1, 4, 6, 9], [4, 6, 7, 8, 9], [5, 6, 7, 8, 9], [4, 5], 
                           [1, 4, 6, 8, 9], [4, 7, 8, 9], [2, 4, 5, 9], [2, 3, 6, 7], [1, 3, 4, 5, 6, 8], 
                           [2, 3, 4, 5, 7], [3, 4, 5, 8], [4, 5], [1, 2, 3, 6, 7, 8, 9]]

    train_dl, val_dl = get_dataloader(dataset, vfl_num_feature)
    
    # participants = [random_participants]

    # for participant in participants:
    #     count_participant = 1

    #     for p in participant:

    #         begin = time.time()
    #         jmi = joint_mutual_information(dataset=dataset, vfl_num_feature=vfl_num_feature, n_splits=n_splits, participants=p)
    #         end = time.time()
    #         f = open('Records/{}_Records/seed_{}/JMI_seed_{}.txt'.format(dataset, random_seed, random_seed), 'a')
    #         f.write('{}'.format(jmi))
    #         f.write('\n')

    #         print('Total time needed for JMI Computation is {}'.format(end - begin))

    #         count_participant += 1

    if dataset == 'mnist':
        greedy_participants = [[1,8,9,3,2], [1,2,9,4,6], [1,4,6,2,3], [1,4,3,2,9], [1,2,5,8,4], [1,3,8,7,9]]
        nips_participants = [[8,6,3,1,7], [6,7,3,8,1], [6,1,3,7,8], [3,6,7,8,1], [3,7,6,2,1], [3,6,7,1,8]]

        greedy_participants_2 = [[1,8,9], [1,2,9], [1,4,6], [1,4,3], [1,2,5], [1,3,8]]
        nips_participants_2 = [[8,6,1], [6,7,1], [6,1,3], [3,6,1], [3,7,1], [3,6,1]]

    elif dataset == 'fashion_mnist':
        greedy_participants = [[1,3,9,8,6], [1,8,4,5,9], [1,2,7,3,9], [1,7,8,4,6], [1,9,3,8,7], [1,9,7,2,5]]
        nips_participants = [[1,9,5,4,8], [1,9,5,4,8], [1,9,2,7,8], [1,9,8,7,6], [1,9,6,3,8], [1,9,2,5,4]]

        greedy_participants_2 = [[1,3,9], [1,8,4], [1,2,7], [1,7,8], [1,9,3], [1,9,7]]
        nips_participants_2 = [[1,9,5], [1,9,5], [1,9,2], [1,9,8], [1,9,6], [1,9,2]]

    elif dataset == 'COIL20':
        greedy_participants = [[1,2,7,3,4], [1,2,7,8,3], [1,9,3,8,2], [1,4,6,8,7], [1,4,9,5,7], [1,6,7,4,5]]
        nips_participants = [[1,9,5,2,4], [1,9,5,2,4], [1,9,5,3,2], [1,9,4,2,6], [1,9,4,5,8], [1,9,6,4,7]]

        greedy_participants_2 = [[1,2,7], [1,2,7], [1,9,3], [1,4,6], [1,4,9], [1,6,7]]
        nips_participants_2 = [[1,9,5], [1,9,5], [1,9,5], [1,9,4], [1,9,4], [1,9,6]]

    elif dataset == 'Isolet':
        greedy_participants = [[1,8,6,3,9], [1,2,4,5,7], [1,8,3,6,4], [1,7,8,3,5], [1,9,3,4,8], [1,7,3,2,8]]
        nips_participants = [[1,8,3,6,7], [1,7,2,8,6], [1,3,6,8,7], [1,7,8,3,9], [1,9,8,4,6], [1,7,3,2,8]]

        greedy_participants_3 = [[1,8,6,3], [1,2,4,5], [1,8,3,6], [1,7,8,3], [1,9,3,4], [1,7,3,2]]
        nips_participants_3 = [[1,8,3,6], [1,7,2,8], [1,3,6,8], [1,7,8,3], [1,9,8,4], [1,7,3,2]]

        greedy_participants_2 = [[1,8,6], [1,2,4], [1,8,3], [1,7,8], [1,9,3], [1,7,3]]
        nips_participants_2 = [[1,8,3], [1,7,2], [1,3,6], [1,7,8], [1,9,8], [1,7,3]]


    # GREEDY
    jmi_log = []
    time_log = []

    begin = time.time()
    jmi = joint_mutual_information(dataset=dataset, vfl_num_feature=vfl_num_feature, n_splits=n_splits, participants=greedy_participants[random_seed])
    end = time.time()

    jmi_log.append(jmi)
    time_log.append(end - begin)

    print('Total time needed for JMI Computation is {}'.format(end - begin))

    results = pd.DataFrame(
            {
            "jmi":jmi_log,
            "time": time_log, 
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/greedy_jmi_{}_4.csv'.format(dataset, random_seed, random_seed))


    # NIPS
    jmi_log = []
    time_log = []

    begin = time.time()
    jmi = joint_mutual_information(dataset=dataset, vfl_num_feature=vfl_num_feature, n_splits=n_splits, participants=nips_participants[random_seed])
    end = time.time()

    jmi_log.append(jmi)
    time_log.append(end - begin)

    print('Total time needed for JMI Computation is {}'.format(end - begin))

    results = pd.DataFrame(
            {
            "jmi":jmi_log,
            "time": time_log, 
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/nips_jmi_{}_4.csv'.format(dataset, random_seed, random_seed))


    # 2 participants
    # GREEDY
    jmi_log = []
    time_log = []

    begin = time.time()
    jmi = joint_mutual_information(dataset=dataset, vfl_num_feature=vfl_num_feature, n_splits=n_splits, participants=greedy_participants_2[random_seed])
    end = time.time()

    jmi_log.append(jmi)
    time_log.append(end - begin)

    print('Total time needed for JMI Computation is {}'.format(end - begin))

    results = pd.DataFrame(
            {
            "jmi":jmi_log,
            "time": time_log, 
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/greedy_jmi_{}_2.csv'.format(dataset, random_seed, random_seed))


    # NIPS
    jmi_log = []
    time_log = []

    begin = time.time()
    jmi = joint_mutual_information(dataset=dataset, vfl_num_feature=vfl_num_feature, n_splits=n_splits, participants=nips_participants_2[random_seed])
    end = time.time()

    jmi_log.append(jmi)
    time_log.append(end - begin)

    print('Total time needed for JMI Computation is {}'.format(end - begin))

    results = pd.DataFrame(
            {
            "jmi":jmi_log,
            "time": time_log, 
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/nips_jmi_{}_2.csv'.format(dataset, random_seed, random_seed))

    if dataset == 'Isolet':
        # 3 participants
        # GREEDY
        jmi_log = []
        time_log = []

        begin = time.time()
        jmi = joint_mutual_information(dataset=dataset, vfl_num_feature=vfl_num_feature, n_splits=n_splits, participants=greedy_participants_3[random_seed])
        end = time.time()

        jmi_log.append(jmi)
        time_log.append(end - begin)

        print('Total time needed for JMI Computation is {}'.format(end - begin))

        results = pd.DataFrame(
            {
            "jmi":jmi_log,
            "time": time_log, 
            }
            )
        results.head()

        results.to_csv('Records/{}_Records/seed_{}/greedy_jmi_{}_3.csv'.format(dataset, random_seed, random_seed))


        # NIPS
        jmi_log = []
        time_log = []

        begin = time.time()
        jmi = joint_mutual_information(dataset=dataset, vfl_num_feature=vfl_num_feature, n_splits=n_splits, participants=nips_participants_3[random_seed])
        end = time.time()

        jmi_log.append(jmi)
        time_log.append(end - begin)

        print('Total time needed for JMI Computation is {}'.format(end - begin))

        results = pd.DataFrame(
            {
            "jmi":jmi_log,
            "time": time_log, 
            }
            )
        results.head()

        results.to_csv('Records/{}_Records/seed_{}/nips_jmi_{}_3.csv'.format(dataset, random_seed, random_seed))
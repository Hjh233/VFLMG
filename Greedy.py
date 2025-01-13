import argparse
import random
import time

import numpy as np
import pandas as pd
import torch 

import mixed
from preprocess import LoadData, PrepareSimpleVFLDataLoader, PrepareSimpleVFLDataset


total_epoch = 50
learning_rate = 0.003
hidden_dim=[50, 10]
bottom_model_dim = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  



def get_dataloader(dataset, vfl_num_feature):
    x_train, y_train = LoadData('{}_train'.format(dataset))
    x_test, y_test = LoadData('{}_test'.format(dataset))

    train_loader, val_loader = PrepareSimpleVFLDataLoader(x_train, y_train, x_test, y_test, vfl_num_feature)

    return train_loader, val_loader



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



if __name__ == '__main__':

    import copy

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

    original_participant = [2,3,4,5,6,7,8,9]
    greedy_participant = [1]
 
    candidate_participants = [[2], [3], [4], [5], [6], [7], [8], [9]]

    jmi_log = []
    time_log = []

    train_dl, val_dl = get_dataloader(dataset, vfl_num_feature)

    for i in range(7):
        
        score = np.zeros(8 - i)

        count = 0
        for p in candidate_participants:
            begin = time.time()
            score[count] = joint_mutual_information(dataset=dataset, vfl_num_feature=vfl_num_feature, n_splits=n_splits, participants=p)
            end = time.time()

            print('JMI computaion time is', end - begin)

            jmi_log.append(score[count])
            time_log.append(end - begin)

            count += 1
        
        index = np.argmax(score) 
        chosen_participant = original_participant[index]

        greedy_participant.append(chosen_participant)
        print('greedy participant is', greedy_participant)

        original_participant.remove(chosen_participant)

        candidate_participants = []

        for left_p in original_participant:
            temp_greedy_participant = copy.deepcopy(greedy_participant)
            temp_greedy_participant.append(left_p)
            candidate_participants.append(temp_greedy_participant)

    greedy_participant.append(original_participant[0])

    f = open('Records/{}_Records/seed_{}/greedy_sort_{}.csv'.format(dataset, random_seed, random_seed), "w")

    for item in greedy_participant:
        f.write(str(item))

    f.close()

    results = pd.DataFrame(
            {
            "jmi":jmi_log,
            "time": time_log, 
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/greedy_log_{}.csv'.format(dataset, random_seed, random_seed))
  

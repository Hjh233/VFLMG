import argparse
import random

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch 
from torch import nn

from preprocess import get_dataloader


total_epoch = 50
learning_rate = 0.003
hidden_dim=[100, 26]
bottom_model_dim = hidden_dim[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  



class BottomModel(nn.Module):
    def __init__(self, backbone):
      super().__init__()
      if not isinstance(backbone, nn.Module):
        raise TypeError('Backbone network should be an instance of nn.Module')
      self.backbone = backbone
  
    def forward(self, x):
      outputs = self.backbone(x)
      return outputs



class MultipleClientVNN(nn.Module):
    def __init__(self, clients_btm, clients_dim, hidden_dim):
        '''
            Params:
            clients_btm: List of BottomModel instances
            clients_dim: List of output_dims of clients
        '''
        super().__init__()

        for client in clients_btm:
            if not isinstance(client, BottomModel):
                raise TypeError('Bottom model should be an instance of BottomModel')
            
        self.clients_btm = clients_btm
        self.clients_dim = clients_dim
        self.emb_dim = clients_dim[0]

        modules = []
        modules.append(nn.Sequential(
            nn.Linear(len(clients_dim) * bottom_model_dim, hidden_dim[0]), 
            nn.ReLU(True)))
        for idx in range(1, len(hidden_dim)-1):
            modules.append(nn.Linear(hidden_dim[idx-1], hidden_dim[idx]))
            modules.append(nn.ReLU(True))
        # Lastlayer without ReLU
        modules.append(nn.Linear(hidden_dim[-2], hidden_dim[-1]))
        self.top_model = nn.Sequential(*modules)


    def forward(self, input_x):
        '''
            input_x: List of input data
        '''
        count = 0

        for client in self.clients_btm:
            if count == 0:
                emb = client(input_x[0])
            else:
                emb = torch.cat((emb, client(input_x[count])), 1)
            count += 1

        out = self.top_model(emb)
        # return torch.sigmoid(out)

        return out
    

'''
   Model Training and Testing
'''

def joint_training(train_loader, val_loader, vfl_num_feature, n_splits, participants):
    client_feature_dim = vfl_num_feature // n_splits
    clients_btm = []
    clients_dim = []

    for i in range(len(participants)):
        # MNIST
        client = nn.Sequential(nn.Linear(client_feature_dim, 100), nn.ReLU(True), 
                          nn.Linear(100, bottom_model_dim), nn.ReLU(True))

        # criteo
        # client = nn.Sequential(nn.Linear(client_feature_dim, 40), nn.ReLU(True), 
        #                        nn.Linear(40, 20), nn.ReLU(True),
        #                        nn.Linear(20, 5), nn.ReLU(True),
        #                        nn.Linear(5, 4), nn.ReLU(True),)

        client = BottomModel(client)
        client = client.to(device)

        clients_btm.append(client)
        clients_dim.append(client_feature_dim)


    vfl = MultipleClientVNN(clients_btm=clients_btm, clients_dim=clients_dim, hidden_dim=hidden_dim)
    vfl = vfl.to(device)

    print('Vertical NN model structure', vfl)

    loss_log = []
    train_acc_log = []
    train_auc_log = []
    val_acc_log = []
    val_auc_log = []

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(vfl.parameters(), lr=0.003)
    epochs = total_epoch
    for e in range(epochs):
        train_acc = 0
        train_loss = 0
        total_train = 0

        val_acc = 0  
        total_val = 0

        step = vfl_num_feature // n_splits
        train_labels, train_probs = np.array([]), np.array([])
        val_labels, val_probs = np.array([]), np.array([])

        for data, labels in train_loader:
    
            data, labels = data.float().to(device), labels.float().to(device)

            clients_data = []

            for participant in participants:
                begin_idx = (participant - 1) * step
                if participant != n_splits:
                    end_idx = participant * step
                else:
                    end_idx = vfl_num_feature

                clients_data.append(data[:, begin_idx:end_idx])

            optimizer.zero_grad()
            out = vfl(clients_data)
            out = out.to(device)

            train_labels = np.append(train_labels, labels.cpu().numpy().astype(np.int32))
            train_probs = np.append(train_probs, torch.sigmoid(out[:, 1]).detach().cpu().numpy())
                                        
            labels = labels.type(torch.LongTensor).to(device)
            loss = criterion(out, labels)

            _, predicted = torch.max(out.data, 1)

            train_acc += (predicted == labels).sum().item()
            total_train += labels.size(0)
       
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        n_classes = len(np.unique(train_labels))
        if n_classes == 2:
            train_auc = roc_auc_score(train_labels, train_probs)
        else:
            train_auc = 0

        vfl.eval()
        for data, labels in val_loader:
    
            data, labels = data.float().to(device), labels.float().to(device)

            clients_data = []

            for participant in participants:
                begin_idx = (participant - 1) * step
                if participant != n_splits:
                    end_idx = participant * step
                else:
                    end_idx = vfl_num_feature

                clients_data.append(data[:, begin_idx:end_idx])

            val_out = vfl(clients_data)
            val_out = val_out.to(device)

            val_labels = np.append(val_labels, labels.cpu().numpy().astype(np.int32))
            val_probs = np.append(val_probs, torch.sigmoid(val_out[:, 1]).detach().cpu().numpy())

            labels = labels.type(torch.LongTensor).to(device)
            _, val_predicted = torch.max(val_out.data, 1)

            val_acc += (val_predicted == labels).sum().item()

            total_val += labels.size(0)

        n_classes = len(np.unique(val_labels))
        if n_classes == 2:
            val_auc = roc_auc_score(val_labels, val_probs)
        else:
            val_auc = 0

        print('*' * 100)
        print(' Training accuracy in epoch {} is {}'.format(e + 1, train_acc / total_train))
        print(' Val accuracy in epoch {} is {}'.format(e + 1, val_acc / total_val))
        print(' Training auc in epoch {} is {}'.format(e + 1, train_auc))
        print(' Val auc in epoch {} is {}'.format(e + 1, val_auc))
        print('\n')

        loss_log.append(train_loss/len(train_loader))
        train_acc_log.append(train_acc / total_train * 100)
        train_auc_log.append(train_auc)
        val_acc_log.append(val_acc / total_val * 100)
        val_auc_log.append(val_auc)

    return loss_log, train_acc_log, val_acc_log, train_auc_log, val_auc_log




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
 
    print('dataset is {}'.format(dataset))
    train_dl, val_dl = get_dataloader(dataset, vfl_num_feature)

    random_participants = [[[2, 7, 8, 9, 1], [5, 6, 7, 8, 1], [2, 5, 7, 8, 1], [6, 7, 8, 9, 1], [2, 5, 6, 7, 1]],
                           [[2, 6, 7, 9, 1], [4, 5, 7, 9, 1], [2, 3, 6, 9, 1], [2, 5, 7, 9, 1], [3, 5, 8, 9, 1]],
                           [[2, 3, 8, 9, 1], [2, 7, 8, 9, 1], [2, 4, 6, 9, 1], [2, 5, 6, 8, 1], [2, 3, 4, 8, 1]],
                           [[2, 4, 5, 8, 1], [2, 3, 4, 5, 1], [2, 4, 7, 8, 1], [2, 7, 8, 9, 1], [3, 4, 5, 8, 1]],
                           [[2, 4, 7, 8, 1], [4, 5, 6, 8, 1], [2, 3, 5, 8, 1], [3, 5, 7, 9, 1], [3, 4, 6, 9, 1]],
                           [[2, 3, 7, 8, 1], [2, 5, 7, 8, 1], [2, 3, 5, 8, 1], [2, 3, 4, 5, 1], [2, 4, 7, 8, 1]]  
                          ]

    # RANDOM
    random_participant = random_participants[random_seed]

    count = 1

    for participant in random_participant:

        loss_log_baseline, train_acc_log_baseline, val_acc_log_baseline, train_auc_log_baseline, val_auc_log_baseline = joint_training(train_dl, val_dl, vfl_num_feature, n_splits=n_splits, participants=participant)

        results = pd.DataFrame(
            {
            "vfl_loss":loss_log_baseline,
            "vfl_train_acc": train_acc_log_baseline, 
            "vfl_test_acc": val_acc_log_baseline,
            "vfl_train_auc": train_auc_log_baseline, 
            "vfl_test_auc": val_auc_log_baseline,
            }
            )
        results.head()

        results.to_csv('Records/{}_Records/seed_{}/random_{}_{}.csv'.format(dataset, random_seed, random_seed, count))

        count += 1


    if dataset == 'mnist':
        greedy_participants = [[1,8,9,3,2], [1,2,9,4,6], [1,4,6,2,3], [1,4,3,2,9], [1,2,5,8,4], [1,3,8,7,9]]
        nips_participants = [[8,6,3,1,7], [6,7,3,8,1], [6,1,3,7,8], [3,6,7,8,1], [3,7,6,2,1], [3,6,7,1,8]]

    elif dataset == 'fashion_mnist':
        greedy_participants = [[1,3,9,8,6], [1,8,4,5,9], [1,2,7,3,9], [1,7,8,4,6], [1,9,3,8,7], [1,9,7,2,5]]
        nips_participants = [[1,9,5,4,8], [1,9,5,4,8], [1,9,2,7,8], [1,9,8,7,6], [1,9,6,3,8], [1,9,2,5,4]]

    # NIPS
    loss_log_baseline, train_acc_log_baseline, val_acc_log_baseline, train_auc_log_baseline, val_auc_log_baseline = joint_training(train_dl, val_dl, vfl_num_feature, n_splits=n_splits, participants=nips_participants[random_seed])

    results = pd.DataFrame(
            {
            "vfl_loss":loss_log_baseline,
            "vfl_train_acc": train_acc_log_baseline, 
            "vfl_test_acc": val_acc_log_baseline,
            "vfl_train_auc": train_auc_log_baseline, 
            "vfl_test_auc": val_auc_log_baseline,
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/nips_{}.csv'.format(dataset, random_seed, random_seed))

    # GREEDY
    loss_log_baseline, train_acc_log_baseline, val_acc_log_baseline, train_auc_log_baseline, val_auc_log_baseline = joint_training(train_dl, val_dl, vfl_num_feature, n_splits=n_splits, participants=greedy_participants[random_seed])

    results = pd.DataFrame(
            {
            "vfl_loss":loss_log_baseline,
            "vfl_train_acc": train_acc_log_baseline, 
            "vfl_test_acc": val_acc_log_baseline,
            "vfl_train_auc": train_auc_log_baseline, 
            "vfl_test_auc": val_auc_log_baseline,
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/greedy_{}.csv'.format(dataset, random_seed, random_seed))

    # 2 participants
    participants_2 = [1]
    for k in range(2):
        participants_2.append(greedy_participants[random_seed][k+1])

    loss_log_baseline, train_acc_log_baseline, val_acc_log_baseline, train_auc_log_baseline, val_auc_log_baseline = joint_training(train_dl, val_dl, vfl_num_feature, n_splits=n_splits, participants=participants_2)

    results = pd.DataFrame(
            {
            "vfl_loss":loss_log_baseline,
            "vfl_train_acc": train_acc_log_baseline, 
            "vfl_test_acc": val_acc_log_baseline,
            "vfl_train_auc": train_auc_log_baseline, 
            "vfl_test_auc": val_auc_log_baseline,
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/2_passive_parties_{}.csv'.format(dataset, random_seed, random_seed))

    # FULL
    loss_log_baseline, train_acc_log_baseline, val_acc_log_baseline, train_auc_log_baseline, val_auc_log_baseline = joint_training(train_dl, val_dl, vfl_num_feature, n_splits=n_splits, participants=[1,2,3,4,5,6,7,8,9])

    results = pd.DataFrame(
            {
            "vfl_loss":loss_log_baseline,
            "vfl_train_acc": train_acc_log_baseline, 
            "vfl_test_acc": val_acc_log_baseline,
            "vfl_train_auc": train_auc_log_baseline, 
            "vfl_test_auc": val_auc_log_baseline,
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/full_training_{}.csv'.format(dataset, random_seed, random_seed))

    # ALONE
    loss_log_baseline, train_acc_log_baseline, val_acc_log_baseline, train_auc_log_baseline, val_auc_log_baseline = joint_training(train_dl, val_dl, vfl_num_feature, n_splits=n_splits, participants=[1])

    results = pd.DataFrame(
            {
            "vfl_loss":loss_log_baseline,
            "vfl_train_acc": train_acc_log_baseline, 
            "vfl_test_acc": val_acc_log_baseline,
            "vfl_train_auc": train_auc_log_baseline, 
            "vfl_test_auc": val_auc_log_baseline,
            }
            )
    results.head()

    results.to_csv('Records/{}_Records/seed_{}/alone_{}.csv'.format(dataset, random_seed, random_seed))
    
import copy
import os
import random
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, DataLoader


'''
   Hyperparameter setting
'''
batch_size = 100
query_instance = 10000
trainset_ratio = 0.8
total_participants = 9
arg_dataset = sys.argv[2]
arg_seed = int(sys.argv[6])
print('arg_seed', arg_seed)


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  

setup_seed(arg_seed)


'''Loading .mat data'''

def LoadData(dataset):
    if 'mnist' in dataset and 'fashion' not in dataset:
        import mnist
        x_train, y_train, x_test, y_test = mnist.load()

    elif 'fashion_mnist' in dataset:
        import dataget

        x_train, y_train, x_test, y_test = dataget.image.fashion_mnist().get()

        x_train = x_train.reshape(60000, -1)
        x_test = x_test.reshape(10000, -1)

    elif 'criteo' in dataset:
        criteo_train, criteo_test = pd.read_csv('data/tabular/criteo_train.csv', header=None), pd.read_csv('data/tabular/criteo_test.csv', header=None)
        criteo_train, criteo_test = criteo_train.to_numpy(), criteo_test.to_numpy()
        x_train, x_test = criteo_train[:, 2:], criteo_test[:, 2:]
        y_train, y_test = criteo_train[:, 1], criteo_test[:, 1]

    elif 'gisette' in dataset or 'COIL20' in dataset or 'Isolet' in dataset:
        import itertools

        name = "".join(itertools.takewhile(lambda x: x!="_", dataset))

        file_name = '{}.mat'.format(name)
        mat = loadmat(os.path.join('data/mat', file_name))
        x = mat["X"]
        y = mat["Y"]

        y = y.flatten()
        if 'gisette' in dataset:
            y[np.where(y == -1)] = 0   
        elif 'COIL20' in dataset or 'Isolet' in dataset:
            y -= 1

        total_samples = x.shape[0]

        setup_seed(arg_seed)

        all_samples = np.random.choice(total_samples, total_samples, replace = False)
        random_samples = np.random.choice(total_samples, int(total_samples * trainset_ratio), replace = False)
        complementary_samples = np.array(list(set(all_samples)-set(random_samples)))

        x_train, x_test = x[random_samples, :], x[complementary_samples, :]
        y_train, y_test = y[random_samples], y[complementary_samples]


    if 'train' in dataset:
        X = copy.deepcopy(x_train)
        X = X.astype(np.float32)
        Y = copy.deepcopy(y_train)

    if 'test' in dataset:
        X = copy.deepcopy(x_test)
        X = X.astype(np.float32)
        Y = copy.deepcopy(y_test)

    num_feature = X.shape[1]
    num_instance = Y.shape[0]

    for i in range(num_feature):
        if np.max(X[:, i]) != 0:
            X[:, i] = (X[:, i] - np.min(X[:, i]))/(np.max(X[:, i]) - np.min(X[:, i]))

    Y = Y.reshape((num_instance))

    return X, Y



class PrepareSimpleVFLDataset(Dataset):

    # load the dataset
    def __init__(self, X_data, y_data, vfl_num_feature, sampling = True):

        total_feature = X_data.shape[1]
        total_instance = X_data.shape[0]
        print('total instance', total_instance)

        setup_seed(arg_seed)

        random_feature = np.random.choice(total_feature, vfl_num_feature, replace=False)
        if arg_seed == 0 or arg_seed == 1 or arg_seed == 2:
            feature_per_party = int(vfl_num_feature / total_participants)
            random_feature[int(1 * feature_per_party):int(2 * feature_per_party)] = random_feature[int(6 * feature_per_party):int(7 * feature_per_party)]
            random_feature[int(3 * feature_per_party):int(4 * feature_per_party)] = random_feature[int(4 * feature_per_party):int(5 * feature_per_party)]

        if sampling and (total_instance > query_instance):
            setup_seed(arg_seed)
            # Here we only sample data in train set, not test set
            all_instance = np.random.choice(total_instance, total_instance, replace = False)
            random_instance = np.random.choice(total_instance, query_instance, replace = False)
            print(len(random_instance))

            complementary_instance = np.array(list(set(all_instance)-set(random_instance)))

            X_data_vfl = copy.deepcopy(X_data[random_instance, :])
            y_data_vfl = copy.deepcopy(y_data[random_instance])
        else:
            X_data_vfl = copy.deepcopy(X_data)
            y_data_vfl = copy.deepcopy(y_data)       

        # ensure input data's type is float
        self.input_x = X_data_vfl[:, random_feature]
        self.y = y_data_vfl

        self.input_x = self.input_x.astype('float32')
        self.y = self.y.astype('float32')

        self.random_feature = random_feature

    # number of rows in the dataset
    def __len__(self):
        return len(self.input_x)

    # get one sample of a certain index
    def __getitem__(self, idx):
        return [self.input_x[idx], self.y[idx]]



def PrepareSimpleVFLDataLoader(X_train_data, y_train_data, X_test_data, y_test_data, vfl_num_feature):
    train_dataset = PrepareSimpleVFLDataset(X_train_data, y_train_data, vfl_num_feature)
    test_dataset = PrepareSimpleVFLDataset(X_test_data, y_test_data, vfl_num_feature)

    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dl, test_dl



def get_dataloader(dataset, vfl_num_feature):
    x_train, y_train = LoadData('{}_train'.format(dataset))
    x_test, y_test = LoadData('{}_test'.format(dataset))

    train_loader, val_loader = PrepareSimpleVFLDataLoader(x_train, y_train, x_test, y_test, vfl_num_feature)

    return train_loader, val_loader



def main():
    x_train, y_train = LoadData('mnist_train')
    x_test, y_test = LoadData('mnist_test')

    input_size = x_train.shape[1]
    print('sample number:', x_train.shape[0])
    print('feature number:', x_train.shape[1])

    train_loader, test_loader = PrepareSimpleVFLDataLoader(x_train, y_train, x_test, y_test, 784)

    print(len(train_loader.dataset), len(test_loader.dataset))



if __name__ == '__main__':
    x, y = LoadData('mnist_train')
    print(x.shape)
    print(y.shape)


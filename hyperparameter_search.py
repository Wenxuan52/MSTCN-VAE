#!/usr/bin/env python
import argparse
import sys
import random
import copy
import yaml

# torchlight
import torchlight
from torchlight import import_class

# import ST-GCN model
from net.st_gcn import Model

# define function to optimize hyperparameters
def optimize_hyperparams(hyperparams):
    # create new train.yaml file with hyperparameters set to values passed in
    with open('config/st_gcn/imigue/train.yaml', 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    train_config['model_args']['dropout'] = hyperparams['dropout']
    train_config['weight_decay'] = hyperparams['weight_decay']
    train_config['base_lr'] = hyperparams['base_lr']
    train_config['step'] = hyperparams['step']
    with open('config/st_gcn/imigue/train_new.yaml', 'w') as f:
        yaml.dump(train_config, f)

    # train model using new train.yaml file
    Processor = import_class('processor.recognition.REC_Processor')
    p = Processor(['--config', 'config/st_gcn/imigue/train_new.yaml'])
    p.start()

    # return cross-validation accuracy
    return p.best_acc

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    processors['demo_old'] = import_class('processor.demo_old.Demo')
    processors['demo'] = import_class('processor.demo_realtime.DemoRealtime')
    processors['demo_offline'] = import_class('processor.demo_offline.DemoOffline')
    # endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # perform random search to optimize hyperparameters
    search_space = {
        'dropout': [0.1, 0.2, 0.3],
        'weight_decay': [0.00001, 0.0001, 0.001],
        'base_lr': [0.0001, 0.001, 0.01],
        'step': [[5, 10], [10, 20], [20, 40]]
    }
    num_trials = 10
    best_hyperparams = None
    best_acc = 0
    for i in range(num_trials):
        hyperparams = {
            'dropout': random.choice(search_space['dropout']),
            'weight_decay': random.choice(search_space['weight_decay']),
            'base_lr': random.choice(search_space['base_lr']),
            'step': random.choice(search_space['step'])
        }
        acc = optimize_hyperparams(hyperparams)
        print(f'Trial {i+1}/{num_trials}: {hyperparams} -> {acc}')
        if acc > best_acc:
            best_hyperparams = copy.deepcopy(hyperparams)
            best_acc = acc
    print(f'Best hyperparameters: {best_hyperparams} -> {best_acc}')

import os
import matplotlib.pyplot as plt
from DataClass import Data
import torch
from timeit import default_timer as timer
from model import Generator, Discriminator
import torch.nn as nn
from utils import *

if __name__ == '__main__':
    config = {
        "normal_num": 0,
        "ratio": 0.1,
        "batch_size": 65,
        'epoch': 200,
        'learning_rate': 0.0002,
        "experiment_num": 1,
        'use_save_model': False,
        'lambda':0.01,
        'mean_median':'median',
        'normalize' : 'standard',
        'download': False # MNIST를 다운할 것인지
    }
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    data =Data(config)
    train_loader = torch.utils.data.DataLoader(dataset=data.train_dataset,batch_size=config['batch_size'],shuffle=True,drop_last=True)
    plt.imshow(train_loader.dataset.x[0][0],cmap='gray')
    plt.show()
    plt.imshow(train_loader.dataset.x[400][0],cmap='gray')

    start = timer()
    generator = Generator().cuda()
    discriminator = Discriminator(config).cuda()
    print("GPU의 할당시간 : {:4f}".format(timer()-start))

    if config['use_save_model']:
        generator.load_state_dict(torch.load('saved_model/generator.pkl'))
        discriminator.load_state_dict(torch.load('saved_model/discriminator.pkl'))
        print("\n--------model restored--------\n")
    else:
        print("\n--------Not using restored Model--------\n")

    train(config, generator, discriminator, train_loader)
    test(data, generator, discriminator, config['lambda'], config['normal_num'], config['normalize'])
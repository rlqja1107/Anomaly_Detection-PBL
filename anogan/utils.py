import matplotlib.pyplot as plt
from numpy.ma import default_fill_value
import torch
import torch.nn as nn
from timeit import default_timer as timer
import torch.nn.init as init
from torch.autograd import Variable 
import torchvision.utils as vutils
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn import metrics
from math import sqrt 


def image_check(gen_fake):
    img = gen_fake.data.numpy()
    for i in range(2):
        plt.imshow(img[i][0],cmap='gray')
        plt.show()

def Anomaly_score(x,G_z, discriminator, Lambda=0.1):
    _,x_feature = discriminator(x)
    _,G_z_feature = discriminator(G_z)
    
    residual_loss = torch.sum(torch.abs(x-G_z))
    discrimination_loss = torch.sum(torch.abs(x_feature-G_z_feature))
    
    total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss
    return total_loss

def fit_latent_space(z_optimizer, generator, z, test_data, discriminator, lambdas):
    start = timer()
    for i in range(5000):
        z_optimizer.zero_grad()
        gen_fake = generator(z)
        loss = Anomaly_score(test_data,gen_fake,discriminator,Lambda=lambdas)
        loss.backward()
        z_optimizer.step()
    
        if i%1000==0:
            print("{}th Loss : {:4f}".format(i,loss.cpu().data))
    print("Fit Latent Space Time : {:4f}".format(timer()-start))
    return z

def normalization(df, standard):
    if standard == 'standard':
        disc_mean = np.mean(df['disc_loss'])
        disc_std = np.std(df['disc_loss'], ddof = 1)
        score_mean = np.mean(df['residual'])
        score_std = np.std(df['residual'], ddof = 1)
        df['residual'] = (df['residual'] - score_mean) / score_std
        df['disc_loss'] = (df['disc_loss'] - disc_mean) / disc_std
    else:
        disc_min = np.min(df['disc_loss'])
        disc_max = np.max(df['disc_loss'])
        score_min = np.min(df['residual'])
        score_max = np.max(df['residual'])
        df['residual'] = (df['residual'] - score_min) / (score_max-score_min)
        df['disc_loss'] = (df['disc_loss'] - disc_min) / (disc_max - disc_min)

def metric_result(df):
    
    ratio = [0,0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999, 1]
    disc_r, res_r, thres,au, re, pre, f1 = [],[],[],[],[],[],[]
    for r in ratio:
        df['ano_score'] = (1-r)* df['residual'] + r*df['disc_loss']
        fpr, tpr, thresholds = metrics.roc_curve(df['y'].tolist(), df['ano_score'].tolist(), pos_label=1)
        dist = []
        min_idx = -1
        min_dist = 1000.0
        for i in range(len(fpr)):
            distance = sqrt((fpr[i]-0.0)**2 + (tpr[i]-1.0)**2)
            dist.append(distance)
            if distance < min_dist:
                min_idx = i
                min_dist = distance
        df['predict_y'] = df['ano_score'].apply(lambda x:1 if x>thresholds[min_idx] else 0)
        TN, FP, FN, TP = confusion_matrix(df['y'].tolist(), df['predict_y'].tolist(), labels = [0,1]).ravel()
        auc = metrics.auc(fpr, tpr)
        eps = 10e-5
        precision = TN/(TN+FN+eps)
        recall = TN/(TN+FP+eps)
        f1score = (2*precision*recall)/(precision+recall)
        disc_r.append(r)
        res_r.append(1-r)
        au.append(auc)
        re.append(recall)
        pre.append(precision)
        f1.append(f1score)
        thres.append(thresholds[min_idx])
    metric = pd.DataFrame(np.asarray([disc_r,res_r, thres, au,re,pre,f1]).T, columns = ['Dis Ratio', 'Residual', 'Threshold','AUC', 'Recall', 'Precision', 'F1'])
    return metric

def print_result(df, metric):
    print("-------------각 Network에서의 Anomaly Score의 평균-------------")
    print("Discrimination의 Anomaly score : {:4f}".format(np.mean(df[df['y']==1]['disc_loss'])))
    print("Discrimination의 Normal score : {:4f}".format(np.mean(df[df['y']==0.0]['disc_loss'])))
    print("Residual Anomaly score : {:4f}".format(np.mean(df[df['y']==1]['residual'])))
    print("Residual Normal score : {:4f}".format(np.mean(df[df['y']==0.0]['residual'])))
    print("-------------Discrimination과 Residual 비율에 따른 metric-------------")
    print(metric)


def test(data, generator, discriminator, lambdas, normal_num, standard):
    z = Variable(init.normal_(torch.zeros(len(data.test_dataset),100).cuda(),mean=0,std=0.1),requires_grad=True)
    z_optimizer = torch.optim.Adam([z],lr=1e-4)
    gen_fake = generator(z)
    test_data =  data.test_dataset.x.view(-1,1,28,28).cuda()

    z = fit_latent_space(z_optimizer, generator, z, test_data, discriminator, lambdas)

    gen_fake = generator(z)

    _,x_feature = discriminator(test_data)
    _,G_z_feature = discriminator(gen_fake)
    test_flatten = test_data.view(test_data.shape[0],-1)
    gen_fake_flatten = gen_fake.view(gen_fake.shape[0],-1)
    resiudal_loss = torch.sum(torch.abs(test_flatten-gen_fake_flatten), axis = 1)
    disc_loss = torch.sum(torch.abs(x_feature-G_z_feature), axis = 1)
    y = data.test_dataset.y.numpy().squeeze(1)
    dat = np.vstack([y,resiudal_loss.detach().cpu().numpy()]).T
    df = pd.DataFrame(dat, columns = ['y', 'residual'])
    df['y'] = df['y'].apply(lambda x:1 if x!=0.0 else 0)
    df['disc_loss'] = disc_loss.detach().cpu().numpy()
    # Normalize
    normalization(df, standard)
    
    metric_df = metric_result(df)

    print_result(df, metric_df)


def train(config, generator, discriminator, train_loader):
    loss_func = nn.MSELoss()

    ones_label = torch.ones(config['batch_size'],1).cuda()
    zeros_label = torch.zeros(config['batch_size'],1).cuda()

    gen_optim = torch.optim.Adam(generator.parameters(), lr= 5*config['learning_rate'],betas=(0.5,0.999))
    dis_optim = torch.optim.Adam(discriminator.parameters(), lr=config['learning_rate'],betas=(0.5,0.999))

    max_loss = 10000000.0
    max_count = 10
    cur_count = 0
    start = timer()
    generator.train()
    discriminator.train()
    for i in range(config['epoch']):
        total_loss = 0.0
        for j,(image,label) in enumerate(train_loader):
            image = image.cuda()
        
        # generator
            gen_optim.zero_grad()
        
            z = init.normal_(torch.Tensor(config['batch_size'],100).cuda(),mean=0,std=0.1)
            gen_fake = generator.forward(z)
            dis_fake,_ = discriminator.forward(gen_fake)
        
            gen_loss = torch.sum(loss_func(dis_fake,ones_label)) # fake classified as real
            gen_loss.backward(retain_graph=True)
            gen_optim.step()
    
        # discriminator
            dis_optim.zero_grad()
        
            z = Variable(init.normal_(torch.Tensor(config['batch_size'],100),mean=0,std=0.1)).cuda()
            gen_fake = generator.forward(z)
            dis_fake,_ = discriminator.forward(gen_fake)
        
            dis_real,_ = discriminator.forward(image)
            dis_loss = torch.sum(loss_func(dis_fake,zeros_label)) + torch.sum(loss_func(dis_real,ones_label))
            dis_loss.backward()
            dis_optim.step()
            total_loss += (gen_loss.detach().item()+ dis_loss.detach().item())
        # model save
            if i % 20 == 0 and j==0:
              #print(gen_loss,dis_loss)
              print("{}th iteration gen_loss: {} dis_loss: {}".format(i,gen_loss.data,dis_loss.data))
            
    
    # Early Stop 용도
        if total_loss < max_loss:
            max_loss = total_loss
            cur_count = 0
        else:
            cur_count += 1
            if cur_count == max_count:
                torch.save(generator.state_dict(),'saved_model/generator.pkl')
                torch.save(discriminator.state_dict(),'saved_model/discriminator.pkl')
                vutils.save_image(gen_fake.data[0:25],"result/gen_{}_{}.png".format(i,j), nrow=5)
                break
      
    print("Train Time : {:4f}".format(timer()-start))
    print("Image Check")
    image_check(gen_fake.cpu())
import matplotlib.pyplot as plt
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

def test(data, generator, discriminator, lambdas, normal_num, mean_median):
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
    total_loss = (1-lambdas)*resiudal_loss + lambdas*disc_loss
    y = data.test_dataset.y.numpy().squeeze(1)
    total_loss = total_loss.detach().cpu().numpy()
    dat = np.vstack([y,total_loss]).T

    df = pd.DataFrame(dat, columns = ['y', 'score'])
    print("Anomaly의 평균 score : {:4f}".format(np.mean(df[df['y']!=normal_num]['score'])))
    print("Normal score의 평균 : {:4f}".format(np.mean(df[df['y']==normal_num]['score'])))
    # Noraml이면 T Abnormal이면 F
    df['y'] = df['y'].apply(lambda x:1 if x==normal_num else 0)

    if mean_median == 'median':
        threshold = np.median(df['score'])
    else:
        threshold = np.mean(df['score'])
    df['predict_y'] = df['score'].apply(lambda x:0 if x>threshold else 1)


    TN, FP, FN, TP = confusion_matrix(df['y'].tolist(), df['predict_y'].tolist()).ravel()
    
    
    print("AUC: {:4f}".format(roc_auc_score(df['y'].tolist(), df['predict_y'].tolist())))
    print("Threshold : {:4f}".format(threshold))
    eps = 10e-5
    precision = TP/(TP+FP+eps)
    recall = TP/(TP+FN+eps)
    f1score = 2*precision*recall/(precision+recall)
    print("Precision : {:4f}, Recall : {:4f}, F1: {:4f}".format(precision, recall, f1score))
    print("---------------------------------------------------------------------")
    print("TP : {}, TN :{}, FP : {}, FN : {}".format(TP,TN,FP,FN))

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
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
import torchvision.transforms as transforms
import torch
import torchvision.datasets as datasets
class Dataset(Dataset):
  def __init__(self, x, y, n):
    self.x = torch.FloatTensor(x).view(n,1,28,28)
    self.y = torch.FloatTensor(y)

  def __len__(self):
    return len(self.x)
  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]


class Data:
  def __init__(self, config):
    self.normal_num = config['normal_num']
    if config['experiment_num'] == 1 or config['experiment_num'] == 3:
      self.ratio = 0.1
    elif config['experiment_num'] == 2 or config['experiment_num'] == 4:
      self.ratio = 0.01
    transform = Compose([ToTensor()])
    self.normal_num = config['normal_num']
    self.train_dataset, self.test_dataset = self.load_data(config['experiment_num'], config)

  def load_data(self, experiment_num, config):
    train_dataset = datasets.MNIST(root="MNIST/processed/training.pt", train=True,  download=config['download'])
    train_x, train_y = train_dataset.data, train_dataset.targets
    test_dataset = datasets.MNIST(root="MNIST/processed/test.pt", train=False,  download=config['download'])
    if experiment_num == 3 or experiment_num == 4:
      train_x, train_y = self.agumentation10(train_x, train_y)
    test_x, test_y = test_dataset.data, test_dataset.targets
    train_x, train_y = self.preprocessing(train_x, train_y, self.normal_num, self.ratio, True)
    test_x, test_y = self.preprocessing(test_x, test_y, self.normal_num, self.ratio, False)
    n = len(train_x)
    n_ = len(test_x)
    return Dataset(train_x, train_y, n), Dataset(test_x, test_y, n_)
 
  def agumentation10(self, x, y):
    trans_x = transforms.functional.affine(x, angle=0, translate=[0, 0], scale=1, shear=0)
    trans_y = y
    for trans in [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1,-1], [2, 0]]:
      temp = transforms.functional.affine(x, angle=0, translate=trans, scale=1, shear=0)
      trans_x = torch.cat([trans_x, temp])
      trans_y = torch.cat([trans_y, y])
    return trans_x, trans_y
   
  def preprocessing(self, x, y, normal_num, ratio, train):
    """
    Train Dataset에는 모두 0인 것만 포함시키기
    Test Dataset에는 0이 0.8개, 나머지는 0.2개를 포함시키기
    """
    x = x/255.0
    N = len(x)
    processing_x = []
    processing_y = []
    for i in range(N):
      if y[i] == normal_num:
        processing_x.append(x[i].reshape(-1).tolist())
        processing_y.append([y[i].tolist()])
    num_normal_data = len(processing_x)
    print("number of normal data: {}".format(num_normal_data))
    i = 0
    while(1):
      if train:
          break
      if len(processing_x) > num_normal_data*(1+self.ratio):
        break
      if y[i] != normal_num:
        processing_x.append(x[i].reshape(-1).tolist())
        processing_y.append([y[i].tolist()])
      i += 1
    print("number of abnormal data: {}".format(len(processing_x)-num_normal_data))
    return processing_x, processing_y
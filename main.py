{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "name": "anogan.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "authorship_tag": "ABX9TyOlUsZmIv786MOaGBR4tiVI",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/rlqja1107/Anomaly_Detection-PBL-/blob/main/main.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HQ7BiRfJKk46"
      },
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "from torch.utils.data import DataLoader, Dataset\n",
        "import torch.nn.init as init\n",
        "from time import time\n",
        "from torchvision.transforms import Compose, ToTensor\n",
        "import torchvision.transforms as transforms\n",
        "from torchvision import datasets\n",
        "import numpy as np\n",
        "import torchvision.utils as v_utils\n",
        "import os \n",
        "import matplotlib.pyplot as plt\n",
        "os.environ[\"CUDA_VISIBLE_DEVICES\"]=\"0\"\n",
        "from timeit import default_timer as timer"
      ],
      "execution_count": 43,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "pmzt1AxzKqv5"
      },
      "source": [
        "class Dataset(Dataset):\n",
        "  def __init__(self, x, y, n):\n",
        "    self.x = torch.FloatTensor(x).view(n,1,28,28)\n",
        "    self.y = torch.FloatTensor(y)\n",
        "\n",
        "  def __len__(self):\n",
        "    return len(self.x)\n",
        "  def __getitem__(self, idx):\n",
        "    return self.x[idx], self.y[idx]\n",
        "\n",
        "\n",
        "class Data:\n",
        "  def __init__(self, config):\n",
        "    transform = Compose([ToTensor()])\n",
        "    self.normal_num = config['normal_num']\n",
        "    self.ratio = config['ratio']\n",
        "    self.train_dataset, self.test_dataset = self.load_data()\n",
        "\n",
        "  def load_data(self):\n",
        "\n",
        "    train_dataset = datasets.MNIST(root=\"MNIST/processed/training.pt\", train=True,  download=False)\n",
        "    train_x, train_y = train_dataset.data, train_dataset.targets\n",
        "    test_dataset = datasets.MNIST(root=\"MNIST/processed/test.pt\", train=False,  download=False)\n",
        "    test_x, test_y = test_dataset.data, test_dataset.targets\n",
        "    train_x, train_y = self.preprocessing(train_x, train_y, self.normal_num, self.ratio, True)\n",
        "    test_x, test_y = self.preprocessing(test_x, test_y, self.normal_num, self.ratio, False)\n",
        "    n = len(train_x)\n",
        "    n_ = len(test_x)\n",
        "    return Dataset(train_x, train_y, n), Dataset(test_x, test_y, n_)\n",
        "\n",
        "  def preprocessing(self, x, y, normal_num, ratio, train):\n",
        "    \"\"\"\n",
        "    Train Dataset에는 모두 0인 것만 포함시키기\n",
        "    Test Dataset에는 0이 0.8개, 나머지는 0.2개를 포함시키기\n",
        "    \"\"\"\n",
        "    x = x/255.0\n",
        "    N = len(x)\n",
        "    processing_x = []\n",
        "    processing_y = []\n",
        "    for i in range(N):\n",
        "      if y[i] == normal_num:\n",
        "        processing_x.append(x[i].reshape(-1).tolist())\n",
        "        processing_y.append([y[i].tolist()])\n",
        "    num_normal_data = len(processing_x)\n",
        "    print(\"number of normal data: {}\".format(num_normal_data))\n",
        "    i = 0\n",
        "    while(1):\n",
        "      if train:\n",
        "          break\n",
        "      if len(processing_x) > num_normal_data*(1+self.ratio):\n",
        "        break\n",
        "      if y[i] != normal_num:\n",
        "        processing_x.append(x[i].reshape(-1).tolist())\n",
        "        processing_y.append([y[i].tolist()])\n",
        "      i += 1\n",
        "    print(\"number of abnormal data: {}\".format(len(processing_x)-num_normal_data))\n",
        "    return processing_x, processing_y"
      ],
      "execution_count": 102,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2LKRkqtMKtVO"
      },
      "source": [
        "config = {\n",
        "    \"normal_num\": 0,\n",
        "    \"ratio\": 0.1,\n",
        "    \"batch_size\": 512,\n",
        "    'threshold': 0.08,\n",
        "    'epoch': 20,\n",
        "    'learning_rate': 0.02\n",
        "}"
      ],
      "execution_count": 112,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "aY_Oa8atT0__"
      },
      "source": [
        ""
      ],
      "execution_count": 103,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Lr2wSLQ-K7XP",
        "outputId": "67c547ed-8b04-4087-aead-2e0062bdfc3c"
      },
      "source": [
        "data =Data(config)\n",
        "train_loader = torch.utils.data.DataLoader(dataset=data.train_dataset,batch_size=config['batch_size'],shuffle=True,drop_last=True)"
      ],
      "execution_count": 104,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "number of normal data: 5923\n",
            "number of abnormal data: 0\n",
            "number of normal data: 980\n",
            "number of abnormal data: 99\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Z8MnYaHjPwBh"
      },
      "source": [
        "\n",
        "# mnist_train = datasets.MNIST(\"MNIST/processed/training.pt\", train=True, \n",
        "#                          transform=transforms.Compose([\n",
        "#                              transforms.ToTensor(),\n",
        "#                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),\n",
        "#                         ]),\n",
        "#                         target_transform=None,\n",
        "#                         download=False)\n",
        "\n",
        "# mnist_test = datasets.MNIST(\"MNIST/processed/test.pt\", train=False, \n",
        "#                          transform=transforms.Compose([\n",
        "#                              transforms.ToTensor(),\n",
        "#                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),\n",
        "#                         ]),\n",
        "#                         target_transform=None,\n",
        "#                         download=False)\n",
        "\n",
        "# # Set Data Loader(input pipeline)\n",
        "\n",
        "# train_loader = torch.utils.data.DataLoader(dataset=mnist_train,batch_size=config['batch_size'],shuffle=True,drop_last=True)"
      ],
      "execution_count": 32,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iXFigOPwOCfk"
      },
      "source": [
        "class Discriminator(nn.Module):\n",
        "    def __init__(self,config):\n",
        "        super(Discriminator,self).__init__()\n",
        "        self.layer1 = nn.Sequential(\n",
        "                        nn.Conv2d(1,8,3,padding=1),   # batch x 16 x 28 x 28\n",
        "                        nn.BatchNorm2d(8),    \n",
        "                        nn.LeakyReLU(),\n",
        "                        nn.Conv2d(8,16,3,stride=2,padding=1),  # batch x 32 x 28 x 28\n",
        "                        nn.BatchNorm2d(16),    \n",
        "                        nn.LeakyReLU(),\n",
        "                        #('max1',nn.MaxPool2d(2,2))   # batch x 32 x 14 x 14\n",
        "        )\n",
        "        self.layer2 = nn.Sequential(\n",
        "                        nn.Conv2d(16,32,3,stride=2,padding=1),  # batch x 64 x 14 x 14\n",
        "                        nn.BatchNorm2d(32),\n",
        "                        nn.LeakyReLU(),\n",
        "                        #nn.MaxPool2d(2,2),\n",
        "                        nn.Conv2d(32,64,3,padding=1),  # batch x 128 x 7 x 7\n",
        "                        nn.BatchNorm2d(64),\n",
        "                        nn.LeakyReLU()\n",
        "        )\n",
        "        self.fc = nn.Sequential(\n",
        "                        nn.Linear(64*7*7,1),\n",
        "                        nn.Sigmoid()\n",
        "        )\n",
        "\n",
        "    def forward(self,x):\n",
        "        out = self.layer1(x)\n",
        "        out = self.layer2(out)\n",
        "        out = out.view(out.size()[0], -1)\n",
        "        feature = out\n",
        "        out = self.fc(out)\n",
        "        return out,feature\n",
        "        \n",
        "class Generator(nn.Module):\n",
        "    def __init__(self):\n",
        "        super(Generator,self).__init__()\n",
        "        self.layer1 = nn.Sequential(\n",
        "             nn.Linear(100,7*7*512),\n",
        "             nn.BatchNorm1d(7*7*512),\n",
        "             nn.ReLU(),\n",
        "        )\n",
        "        self.layer2 = nn.Sequential(\n",
        "                        nn.ConvTranspose2d(512,256,3,2,1,1),\n",
        "                        nn.BatchNorm2d(256),\n",
        "                        nn.LeakyReLU(),\n",
        "                        nn.ConvTranspose2d(256,128,3,1,1),\n",
        "                        nn.BatchNorm2d(128),    \n",
        "                        nn.LeakyReLU(),\n",
        "            )\n",
        "        self.layer3 = nn.Sequential(\n",
        "                        nn.ConvTranspose2d(128,64,3,1,1),\n",
        "                        nn.BatchNorm2d(64),    \n",
        "                        nn.LeakyReLU(),\n",
        "                        nn.ConvTranspose2d(64,1,3,2,1,1),\n",
        "                        nn.Tanh()\n",
        "            )\n",
        "\n",
        "    def forward(self,z):\n",
        "        out = self.layer1(z)\n",
        "        out = out.view(out.size()[0],512,7,7)\n",
        "        out = self.layer2(out)\n",
        "        out = self.layer3(out)\n",
        "        return out\n"
      ],
      "execution_count": 105,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "vXBdxk2kK9Pp",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "a6d78785-057c-4f75-ef62-4eea6f9c1570"
      },
      "source": [
        "start = timer()\n",
        "generator = Generator().cuda()\n",
        "discriminator = Discriminator(config).cuda()\n",
        "print(\"GPU의 할당시간 : {:4f}\".format(timer()-start))\n",
        "try:\n",
        "    generator.load_state_dict(torch.load('./saved_model/generator.pkl'))\n",
        "    discriminator.load_state_dict(torch.load('./saved_model/discriminator.pkl'))\n",
        "    print(\"\\n--------model restored--------\\n\")\n",
        "except:\n",
        "    print(\"\\n--------model not restored--------\\n\")\n",
        "    pass"
      ],
      "execution_count": 106,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "GPU의 할당시간 : 0.043779\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uDjQgrfjLPWx"
      },
      "source": [
        "loss_func = nn.MSELoss()\n",
        "\n",
        "ones_label = torch.ones(config['batch_size'],1).cuda()\n",
        "zeros_label = torch.zeros(config['batch_size'],1).cuda()\n",
        "\n",
        "\n",
        "gen_optim = torch.optim.Adam(generator.parameters(), lr= 5*config['learning_rate'],betas=(0.5,0.999))\n",
        "dis_optim = torch.optim.Adam(discriminator.parameters(), lr=config['learning_rate'],betas=(0.5,0.999))"
      ],
      "execution_count": 107,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ai5akOpqLRFQ"
      },
      "source": [
        "\n",
        "def image_check(gen_fake):\n",
        "    img = gen_fake.data.numpy()\n",
        "    for i in range(2):\n",
        "        plt.imshow(img[i][0],cmap='gray')\n",
        "        plt.show()"
      ],
      "execution_count": 108,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ewTml2Z1LThF",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 607
        },
        "outputId": "13f325d4-46d1-40e9-d6f0-e92734a86df8"
      },
      "source": [
        "start = timer()\n",
        "for i in range(config['epoch']):\n",
        "    \n",
        "    for j,(image,label) in enumerate(train_loader):\n",
        "        image = image.cuda()\n",
        "        # generator\n",
        "        gen_optim.zero_grad()\n",
        "        \n",
        "        z = init.normal_(torch.Tensor(config['batch_size'],100).cuda(),mean=0,std=0.1)\n",
        "        gen_fake = generator(z)\n",
        "        dis_fake,_ = discriminator(gen_fake)\n",
        "        gen_loss = torch.sum(loss_func(dis_fake,ones_label)) # fake classified as real\n",
        "        gen_loss.backward(retain_graph=True)\n",
        "        gen_optim.step()\n",
        "    \n",
        "        # discriminator\n",
        "        dis_optim.zero_grad()\n",
        "        \n",
        "        z = init.normal_(torch.Tensor(config['batch_size'],100).cuda(),mean=0,std=0.1)\n",
        "        gen_fake = generator(z)\n",
        "        dis_fake,_ = discriminator(gen_fake)\n",
        "        \n",
        "        dis_real,_ = discriminator(image)\n",
        "        dis_loss = torch.sum(loss_func(dis_fake,zeros_label)) + torch.sum(loss_func(dis_real,ones_label))\n",
        "        dis_loss.backward()\n",
        "        dis_optim.step()\n",
        "    \n",
        "       # model save\n",
        "        if i % 5 == 0 and j==0:\n",
        "            #print(gen_loss,dis_loss)\n",
        "            torch.save(generator.state_dict(),'saved_model/generator.pkl')\n",
        "            torch.save(discriminator.state_dict(),'saved_model/discriminator.pkl')\n",
        "\n",
        "\n",
        "            print(\"{}th iteration gen_loss: {} dis_loss: {}\".format(i,gen_loss.data,dis_loss.data))\n",
        "            v_utils.save_image(gen_fake.data[0:25],\"result/gen_{}_{}.png\".format(i,j), nrow=5)\n",
        "print(\"Time : {:4f}\".format(timer()-start))\n",
        "image_check(gen_fake.cpu())"
      ],
      "execution_count": 114,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "0th iteration gen_loss: 0.9999978542327881 dis_loss: 3.268695081715123e-06\n",
            "5th iteration gen_loss: 0.9999974966049194 dis_loss: 1.8988206420544884e-06\n",
            "10th iteration gen_loss: 0.9999969601631165 dis_loss: 2.199751406806172e-06\n",
            "15th iteration gen_loss: 0.9999959468841553 dis_loss: 1.8549657170296996e-06\n",
            "Time : 110.799218\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QbNdmIL8LUMP"
      },
      "source": [
        "def Anomaly_score(x,G_z,Lambda=0.1):\n",
        "    _,x_feature = discriminator(x)\n",
        "    _,G_z_feature = discriminator(G_z)\n",
        "    \n",
        "    residual_loss = torch.sum(torch.abs(x-G_z))\n",
        "    discrimination_loss = torch.sum(torch.abs(x_feature-G_z_feature))\n",
        "    \n",
        "    total_loss = (1-Lambda)*residual_loss + Lambda*discrimination_loss\n",
        "    return total_loss"
      ],
      "execution_count": 60,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "131JzO5_LWMK",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8aa52ad0-51c8-4d12-e4c3-8acc4bf3d3d4"
      },
      "source": [
        "z = init.normal(torch.zeros(len(data.test_dataset),100).cuda(),mean=0,std=0.1)\n",
        "z_optimizer = torch.optim.Adam([z],lr=1e-4)\n",
        "\n",
        "gen_fake = generator(z)\n",
        "test_data =  data.test_dataset.x.view(-1,1,28,28).cuda()\n",
        "loss = Anomaly_score(test_data,gen_fake)\n",
        "print(loss)"
      ],
      "execution_count": 115,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "/usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: UserWarning: nn.init.normal is now deprecated in favor of nn.init.normal_.\n",
            "  \"\"\"Entry point for launching an IPython kernel.\n"
          ],
          "name": "stderr"
        },
        {
          "output_type": "stream",
          "text": [
            "tensor(800729.5000, device='cuda:0', grad_fn=<AddBackward0>)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TS4a2-SDX85w",
        "outputId": "76203e39-68ad-4d35-8bf5-b970f3ac6d20"
      },
      "source": [
        "start = timer()\n",
        "for i in range(5000):\n",
        "    gen_fake = generator(z)\n",
        "    loss = Anomaly_score(test_data,gen_fake,Lambda=0.01)\n",
        "    loss.backward()\n",
        "    z_optimizer.step()\n",
        "    \n",
        "    if i%1000==0:\n",
        "        print(loss.cpu().data)\n",
        "print(\"Time : {:4f}\".format(timer()-start))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "tensor(670831.0625)\n",
            "tensor(670831.0625)\n",
            "tensor(670831.0625)\n",
            "tensor(670831.0625)\n",
            "tensor(670831.0625)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Qhl3Nco2dgNm"
      },
      "source": [
        "loss_list = []\n",
        "threshold = 50\n",
        "for i in range(5000):\n",
        "    gen_fake = generator(z)\n",
        "    loss = Anomaly_score(test_data,gen_fake,Lambda=0.01)\n",
        "    loss_list.append(loss.item())"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ul9CHzlvYP40"
      },
      "source": [
        "y = data.test_data.y"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V7aYCzAxd_y8"
      },
      "source": [
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "V_kneyYXWt22"
      },
      "source": [
        "for idx in range(len(test_data)):\n",
        "    target = test_data[idx,0,:,:].numpy()\n",
        "    plt.imshow(target,cmap=\"gray\")\n",
        "    plt.show()\n",
        "    print(\"real data\")\n",
        "\n",
        "    img=gen_fake.cpu().data[idx,0,:,:].numpy()\n",
        "    plt.imshow(img,cmap='gray')\n",
        "    plt.show()\n",
        "    print(\"generated data\")\n",
        "    print(\"\\n------------------------------------\\n\")"
      ],
      "execution_count": 70,
      "outputs": []
    }
  ]
}
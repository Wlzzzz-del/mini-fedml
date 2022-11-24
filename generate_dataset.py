from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import trange
import numpy as np
import random
import json
import os
import torch
import torchvision
import torchvision.transforms as transforms

def read_cifa_data(NUM_USERS=20, NUM_LABELS=3):

    """
    NUM_USERS = 20# 用户数，需要是10的倍数
    NUM_LABELS = 3#  每个设备持有的标签数量
    """
    # 将图像数据转为tensor并且归一化
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    # 读取数据集
    trainset = torchvision.datasets.CIFAR10(root='./data',train = True, download= False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data',train = False, download= False, transform=transform)
    print("read dataset successfully!")
    # 数据加载器
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = len(trainset.data), shuffle = False)
    testloader = torch.utils.data.DataLoader(testset,batch_size = len(testset.data), shuffle = False)

    # 使trainLoader和trainset的顺序保持一致性
    for _,train_data in enumerate(trainloader,0):
        trainset.data, trainset.targets = train_data
    for _,test_data in enumerate(testloader,0):
        testset.data, testset.targets = test_data

    random.seed(1)
    np.random.seed(1)

    # 指定存放数据的路径
    train_path = "./data/train/cifa_train.json"
    test_path = "./data/test/cifa_test.json"
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    cifa_data_image = []
    cifa_data_label = []

    """
    print("shape of trainset:",trainset.data.shape)
    print("this is trainset.data:",trainset.data)
    print("--------------------------------")
    print("this is trainset.data.detach()",trainset.data.detach())
    """
    print("shape of testset:",trainset.targets.shape)

    # 创建一个列表，存放所有的图像数据
    cifa_data_image.extend(trainset.data.cpu().detach().numpy())
    cifa_data_image.extend(testset.data.cpu().detach().numpy())
    cifa_data_label.extend(trainset.targets.cpu().detach().numpy())
    cifa_data_label.extend(testset.targets.cpu().detach().numpy())

    cifa_data_image = np.array(cifa_data_image)
    cifa_data_label = np.array(cifa_data_label)

    # cifa_data按照每个类划分数据集
    """
    总数据集为(60000,3,32,32)，训练集5w张，测试机1w张
    划分为之后每个类有(6000,3,32,32)
    """
    cifa_data = []
    for i in range(10):
        idx = cifa_data_label == i
        cifa_data.append(cifa_data_image[idx])

    print("sample number of each label:\n",[len(v) for v in cifa_data])
    user_labels = []

    ### create user data split ###
    # Assign 30 samples to each user
    X = [[]for _ in range(NUM_USERS)]
    Y = [[]for _ in range(NUM_USERS)]

    # idx用来记录每个类已经被分配的图像

    idx = np.zeros(10,dtype=np.int64)
    for user in range(NUM_USERS):
        for l in range(NUM_LABELS):# 每个客户端有3个标签
            l = (l+user)%NUM_LABELS
            X[user] += (cifa_data[l][idx[l]:idx[l]+10]).tolist()
            Y[user] += (np.ones(10)*l).tolist()
            idx[l] += 10

    # 共20个用户，每个用户分配3类，每类包含10张图像
    print("分配情况：",idx)

    # 服从lognormal分布对每个用户随机分配图像
    user = 0
    props = np.random.lognormal(
        0, 2., (10, NUM_USERS, NUM_LABELS))  # last 5 is 5 labels
    props = np.array([[[len(v)-NUM_USERS]] for v in cifa_data]) * \
        props/np.sum(props, (1, 2), keepdims=True)

    print("分配剩余类别：",props)

    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples) + numran1
            if(NUM_USERS <= 20):
                num_samples = num_samples*2
            if idx[l] + num_samples < len(cifa_data[l]):
                X[user] += cifa_data[l][idx[l]:idx[l]+num_samples].tolist()
                Y[user] += (l*np.ones(num_samples)).tolist()
                idx[l] += num_samples
                print("check len of each user", user, j, "len of data:",len(X[user]), num_samples)

    print("IDX2:", idx)

    # 创建数据集结构
    train_data = {'users':[], 'user_data':{}, 'num_samples':[]}
    test_data = {'users':[],'user_data':{}, 'num_samples':[]}

    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        print(uname)
        # startify 指按照X或者Y的分布分配样本
        X_train, X_test, y_train, y_test = train_test_split(X[i], Y[i], train_size=0.75, stratify=Y[i])

        train_data["user_data"][uname] = {'x':X_train,'y':y_train}
        train_data["users"].append(uname)
        train_data["num_samples"].append(len(X_train))

        test_data["user_data"][uname] = {'x':X_test, 'y':y_test}
        test_data["users"].append(uname)
        test_data["num_samples"].append(len(y_train))

    print("Num_samples train:", train_data['num_samples'])
    print("Num_samples test:", test_data['num_samples'])
    print("total samples:", sum(train_data['num_samples'] + test_data['num_samples']))

    # ---------- write into json--------
    # ----------注释掉即可启用-----------
    ##with open(train_path, 'w') as outfile:
    ##    json.dump(train_data, outfile)
    ##with open(test_path, 'w') as outfile:
    ##    json.dump(test_data, outfile)

    print("Finish Generating Samples!")

    return train_data,test_data
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from tqdm import trange
from scipy.io import loadmat
import numpy as np
import random
import json
import os
import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms

def read_mnist_data(NUM_USERS = 20, NUM_LABELS = 3):
    """
    NUM_USERS = 20# 用户数，需要是10的倍数
    NUM_LABELS = 3#  每个设备持有的标签数量
    """
    PATH = "./Mnist"
    TRAIN_PATH = PATH + "/train"
    TEST_PATH = PATH + "/test"

    # 检查数据集路径是否正确
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)

    if not os.path.exists(TEST_PATH):
        os.makedirs(TEST_PATH)

    # random seed
    np.random.seed(7)
    random.seed(7)

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    # 本地读取Mnist数据集
    Dataset = loadmat(PATH+"/mnist-original.mat")

    X = Dataset["data"].T# 把784改成28*28
    Y = Dataset["label"].T.flatten().astype(np.int64)

    X = torch.tensor(X.reshape((70000,28,28)))
    Y = torch.tensor(Y)
    print(X)
    torch_dataset = Data.TensorDataset(X,Y)
    dataloader = Data.DataLoader(torch_dataset, batch_size=len(torch_dataset),shuffle=False)

    for _,data in enumerate(dataloader,0):
        torch_dataset.data , torch_dataset.target = data

    # 重新构造数据集列表
    mnist_data_image = []
    mnist_data_label = []

    mnist_data_image.extend(torch_dataset.data.cpu().detach().numpy())
    mnist_data_label.extend(torch_dataset.target.cpu().detach().numpy())

    mnist_data_image = np.array(mnist_data_image)
    mnist_data_label = np.array(mnist_data_label)

    # 按照类别划分数据集
    mnist_data = []
    for i in range(10):
        idx = mnist_data_label==i
        mnist_data.append(mnist_data_image[idx])
    print("sample number of each label:\n",[len(v) for v in mnist_data])
    user_labels = []

    ## create user data split ##
    # 分配100个样本给每个用户
    X = [[]for _ in range(NUM_USERS)]
    Y = [[]for _ in range(NUM_USERS)]

    idx = np.zeros(10, dtype=np.int64)
    for user in range(NUM_USERS):
        for l in range(NUM_LABELS):
            l = (l+user)%NUM_LABELS
            X[user] += (mnist_data[l][idx[l]:idx[l]+10]).tolist()
            Y[user] += (np.ones(10)*l).tolist()
            idx[l] += 10

    user = 0
    props = np.random.lognormal(0,2.,(10,NUM_USERS, NUM_LABELS))
    # props = np.array([[[len(v)-NUM_USERS]]] for v in mnist_data) * props/np.sum(props, (1,2), keepdims= True)

    print("分配情况:",idx)

    for user in trange(NUM_USERS):
        for j in range(NUM_LABELS):
            l = (user + j) % 10
            num_samples = int(props[l, user//int(NUM_USERS/10), j])
            numran1 = random.randint(300, 600)
            num_samples = (num_samples) + numran1
            if(NUM_USERS <= 20):
                num_samples = num_samples*2

            if idx[l] + num_samples < len(mnist_data[l]):
                X[user] += mnist_data[l][idx[l]:idx[l] + num_samples].tolist()
                Y[user] += (np.ones(num_samples)*l).tolist()
                idx[l] += num_samples
                print("check len of each user", user, j, "len of data:",len(X[user]), num_samples)
    print("IDX2:", idx)

    # 创建数据集结构
    train_data = {'users':[], 'user_data':{}, 'num_samples':[]}
    test_data = {'users':[], 'user_data':{}, 'num_samples':[]}

    for i in range(NUM_USERS):
        uname = 'f_{0:05d}'.format(i)
        print(uname,"is assigning samples")
        X_train, X_test, y_train, y_test = train_test_split(X[i],Y[i],train_size=0.75, stratify=Y[i])
        train_data["user_data"][uname] = {'x':X_train, 'y':y_train}
        train_data["users"].append(uname)
        train_data["num_samples"].append(len(X_train))

        test_data["user_data"][uname] = {'x':X_test, 'y':y_test}
        test_data["users"].append(uname)
        test_data["num_samples"].append(len(y_train))

    print("Num_samples train:", train_data['num_samples'])
    print("Num_samples test:", test_data['num_samples'])
    print("total samples:", sum(train_data['num_samples'] + test_data['num_samples']))

#------------write into json here --------------
    #with open(TRAIN_PATH, 'w') as outfile:
    #    json.dump(train_data, outfile)
    #with open(TEST_PATH, 'w') as outfile:
    #    json.dump(test_data, outfile)

    print("Finish Generating Samples!")

    return train_data, test_data
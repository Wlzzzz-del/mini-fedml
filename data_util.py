import torch
NUM_CHANNELS_CIFAR = 3
NUM_CHANNELS_MNIST = 1
IMAGE_SIZE_OF_CIFAR = 32
IMAGE_SIZE_OF_MNIST = 28

def read_user_data(id, train_data, test_data, data_name):

    id = train_data["users"][id]
    trainset = train_data["user_data"][id]
    testset = test_data["user_data"][id]

    X_train, Y_train, X_test, Y_test = trainset["x"], trainset["y"], testset["x"], testset["y"]
    if(data_name == "Mnist"):
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_MNIST, IMAGE_SIZE_OF_MNIST, IMAGE_SIZE_OF_MNIST).type(torch.float32)
        Y_train = torch.tensor(Y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_MNIST, IMAGE_SIZE_OF_MNIST, IMAGE_SIZE_OF_MNIST).type(torch.float32)
        Y_test = torch.tensor(Y_test).type(torch.int64)
        pass
    if(data_name == "Cifar10"):
        # view 对tensor 进行reshape， -1表示为自动推导该维度
        X_train = torch.Tensor(X_train).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_OF_CIFAR, IMAGE_SIZE_OF_CIFAR).type(torch.float32)
        Y_train = torch.tensor(Y_train).type(torch.int64)
        X_test = torch.Tensor(X_test).view(-1, NUM_CHANNELS_CIFAR, IMAGE_SIZE_OF_CIFAR, IMAGE_SIZE_OF_CIFAR).type(torch.float32)
        Y_test = torch.tensor(Y_test).type(torch.int64)

    subtrain = [(x, y)for x,y in zip(X_train, Y_train)]
    subtest = [(x, y) for x,y in zip(X_test, Y_test)]

    return id, subtrain, subtest
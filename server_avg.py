from server_base import Server
import data_util
import torch
from generate_dataset import read_cifa_data
from generate_Minist import read_mnist_data
from user_avg import UserAVG
"""
train_data and test_data    data struct:
    train_data = {'users':[], 'user_data':{}, 'num_samples':[]}
    test_data = {'users':[],'user_data':{}, 'num_samples':[]}
"""
class Server_Avg(Server):
    def __init__(self, device, dataset, algorithm, model, batch_size, learning_rate, beta, lamda, num_glob_iters, local_epochs, optimizer, num_users, times):
        super().__init__(device, dataset, algorithm, model[0], batch_size, learning_rate, beta, lamda,num_glob_iters, local_epochs,optimizer,num_users, times)

        # 根据dataset读取不同的数据集,并根据用户数量做切分处理
        if dataset == 'Cifar10':
            train_data ,test_data = read_cifa_data(20, 3)
        if dataset == 'Mnist':
            train_data, test_data = read_mnist_data(20, 3)

        """
            sub_train and sub_test
            data struct:
            sub_train = [(x_1,y_1),...,(x_n,y_n)]
            sub_test = [(x_1,y_1),...,(x_n,y_n)]
        """
        total_users = len(train_data['users'])
        for i in range(total_users):
            id, sub_train, sub_test = data_util.read_user_data(i, train_data, test_data, dataset)
            user = UserAVG(device, id, sub_train, sub_test, model, batch_size, learning_rate, beta, lamda, local_epochs, optimizer)
            self.users.append(user)
            self.total_train_samples += user.train_samples

        print("successfuly create user data")
        print("Number of users/ total_users:",num_users, "/", total_users)
        print("Finishing creating FedAVG server.")

        # 创建用户
        pass


    def train(self):
        loss = []
        for glob_iter in range(self.num_glob_iters):
            print("---------------ROUND NUMBER:",glob_iter,"----------------")
            self.send_parameters()

            self.evaluate()
            self.selected_users = self.select_users(glob_iter, self.num_users)
            for user in self.select_users:
                user.train(self.local_epochs)
            self.aggregate_parameters()
        self.save_model()

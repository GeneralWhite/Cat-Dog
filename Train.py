import torch
import torchvision
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
import torch.nn.functional as f

train_data_path = "training_set/training_set"
test_data_path = "test_set/test_set"

writer = SummaryWriter("logs")
# 图像预处理
img_transforms = transforms.Compose([
    transforms.Resize(size = 256),  # 将所有图片 Resize 为 256 * 256
    transforms.CenterCrop(size = 224),  # 从中央截取为 224 * 224大小
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.4895, 0.4517, 0.4130], std=[0.2587, 0.2506, 0.2514])

])

DataSet = torchvision.datasets.ImageFolder(root = train_data_path, transform = img_transforms)

# 如果图片已经完成分类且储存在对应文件夹，用 ImageFolder 比 Dataset 更方便
# print(len(DataSet))
# print(DataSet)


def get_mean_std(loader):  # 计算均值和方差
    # var[X] = E[X**2] - E[X]**2 方差公式， var[]代表方差，E[]表示期望(平均值)
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0
    # num_batches 为图片数量
    for data, _ in tqdm(loader):  # tqdm 参数为传入可迭代对象
        # _ 不需要 target ,用_表示不需要的值
        channels_sum += torch.mean(data, dim = [1, 2])  # dim 为维度, ？dim的值？
        channels_sqrd_sum += torch.mean(data ** 2, dim = [1, 2])
        num_batches += 1
        # 如果函数接受 DataLoader,即接受 (batch_size, channels, W, H)时将 dim 改为 [0, 2, 3]

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


# mean, std, num_batches = get_mean_std(DataSet)
# print(mean)
# print(std)

def train_val_dataset(dataset, val_split = 0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size = val_split,
                                          random_state = 42)
    #  将原始数据分割为 ”测试集“ 和 "训练集"
    #  第一个参数 array 所要划分的样本结果，可以是列表、numpy数组、scipy稀疏矩阵或pandas的数据框
    #  test_size :样本占比
    #  random_state :设置随机数种子

    datasets = {}
    datasets["train"] = Subset(dataset, train_idx)
    datasets["val"] = Subset(dataset, val_idx)
    # Subset
    # 指定索引处的数据集子集。
    # 参数 dataset 数据集
    # 参数 indices 索引
    return datasets


datasets = train_val_dataset(DataSet)
batch_size = 64
test_data = torchvision.datasets.ImageFolder(test_data_path, img_transforms)
train_data_loader = DataLoader(datasets["train"], batch_size = batch_size)
val_data_loader = DataLoader(datasets["val"], batch_size = batch_size)
test_data_loader = DataLoader(test_data, batch_size = batch_size)

# for imgs, labels in train_data_loader:
#     print(imgs.shape, labels.shape)
#     break


# 以下两块代码为使数据形象化并观察
# def image_convert(img):
#     img = img.clone().cpu().numpy()
#     img = img.transpose(1, 2, 0)
#     std = [ 0.5, 0.5, 0.5 ]
#     mean = [ 0.5, 0.5, 0.5 ]
#     img = img * std + mean
#     return img
#
#
# def plot_10():
#     iter_ = iter(train_data_loader)
#     images, labels = next(iter_)
#     an_ = {'0': 'cat', '1': 'dog'}  # changing labels to be meaningful
#
#     plt.figure(figSize = (20, 10))
#     for idx in range(10):
#         plt.subplot(2, 5, idx + 1)
#         img = image_convert(images[idx])
#         label = labels[idx]
#         plt.imShow(img)
#         plt.title(an_[str(label.numpy())])
#     plt.show()
#
# plot_10()


# 模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size = 3, padding = 0, stride = 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size = 3, padding = 0, stride = 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, padding = 0, stride = 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(3 * 3 * 64, 10)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


simpleNet = SimpleNet()
# 优化器
optimizer = optim.Adam(simpleNet.parameters(), lr = 0.0008)

# device = torch.device("cuda:0")

if torch.cuda.is_available():
    simpleNet = simpleNet.cuda()


def train(model, loss_fn, train_loader, val_loader, epochs = 20):
    total_train_step = 0
    total_test_step = 0
    for epoch in range(0, epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                targets = targets.cuda()
            outputs = model(inputs)
            if torch.cuda.is_available():
                loss_fn = loss_fn.cuda()
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * inputs.size(0)
            total_train_step += 1
            # loss.item()为单个样本平均 loss, inputs.size(0)为 batch_size的大小
            if total_train_step % 10 == 0:
                print("训练次数：{}, Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
        training_loss /= len(train_loader.dataset)
        writer.add_scalar("total_train_loss", training_loss, epoch)
        # training_loss 为一个 epochs 的误差

        model.eval()
        num_correct = 0
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            if torch.cuda.is_available():
                targets = targets.cuda()
                loss_fn = loss_fn.cuda()
            loss = loss_fn(outputs, targets)
            valid_loss += loss.item() * inputs.size(0)
            total_test_step += 1
            correct = torch.eq(torch.max(f.softmax(outputs, dim = 1), dim = 1)[1], targets)
            # targets 为 batch [1] 类型，为了类型一致，要在后面加 [1], ? dim = 1?
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
            # correct 为 1 时正确，全部为 0 ，则为所有数据
        valid_loss /= len(val_loader.dataset)
        total_test_step += 1
        print("valid_Loss:{}".format(valid_loss))
        writer.add_scalar("valid_loss", valid_loss, total_test_step)
        print("accuracy:{}".format(num_correct / num_examples))
        writer.add_scalar("accuracy", num_correct / num_examples, total_test_step)
        torch.save(simpleNet.state_dict(), "simpleNet{}.pth".format(epoch + 1))
        print("模型已保存")


writer.close()
train(simpleNet, loss_fn = torch.nn.CrossEntropyLoss(),
      train_loader = train_data_loader, val_loader = val_data_loader, epochs = 20)

# labels = ['cat', 'dog']
#
# img = Image.open("training_set/training_set/cats/cat.3918.jpg")
#
# img = img_transforms(img).cuda()
# img = torch.unsqueeze(img, 0)
#
# simpleNet.eval()
# prediction = f.softmax(simpleNet(img), dim=1)
# prediction = prediction.argmax()
# print(labels[prediction])
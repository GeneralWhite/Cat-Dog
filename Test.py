import torch
from torch import nn

class Student:
    name = "小卡"
    age = 18

    def __init__(self, aa, bb):
        print("init函数被调用了！")
        print(aa)
        print(bb)

    def fun1(self):
        print("函数1")
        print(self)

    def fun2(self):
        print("函数2")
        print(self)


# Student.fun1(Student(100, 200))
# std = Student(100, 200)
# std.fun1()

input = torch.randn(64, 3, 224, 224)
c1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 7)
c2 = nn.Conv2d(32, 32, 7)
c3 = nn.Conv2d(32, 64, 7)
c4 = nn.Conv2d(64, 64, 7)
c5 = nn.Conv2d(128, 128, 3)
c6 = nn.Conv2d(128, 256, 3)
max = nn.MaxPool2d(kernel_size = 2)
batch = nn.BatchNorm2d(64)
output = c1(input)
output = c2(output)
output = max(output)
output = c3(output)
output = c4(output)
output = max(output)
# output = c5(output)
# output = c6(output)
# output = max(output)
print(output.shape)

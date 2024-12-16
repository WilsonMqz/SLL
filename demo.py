import os

seed = 2

rate_list = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

for rate in rate_list:
    os.system("python main.py --seed {} --dataset fashion --model lenet --rate {}  --lo gce".format(seed, rate))
    os.system("python main.py --seed {} --dataset kuzushiji --model lenet --rate {}  --lo gce".format(seed, rate))
    os.system("python main.py --seed {} --dataset cifar10 --model widenet --rate {}  --lo gce".format(seed, rate))
    os.system("python main.py --seed {} --dataset svhn --model widenet --rate {}  --lo gce".format(seed, rate))

print("all complement!")
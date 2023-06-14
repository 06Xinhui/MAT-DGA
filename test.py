import matplotlib.pyplot as plt
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--person_id', type=int, default=0, help='person index')
args=parser.parse_args()


# x = np.load('./best_accu.npy')
# np.save('./best_accu_{}.npy'.format(args.person_id), x)

total = 0
count_list = [0, 1, 2, 3, 4, 5, 7, 8]
for i in range(10):
    x1 = np.load('./best_accu_{}.npy'.format(i))
    x2 = np.load('./adaptation_only/best_accu_{}.npy'.format(i))
    print(x1)
    if i in count_list:
        total += x1[0]
print(total / 8)


import matplotlib.pyplot as plt
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--person_id', type=int, default=0, help='person index')
args=parser.parse_args()


x = np.load('./best_accu.npy')
np.save('./best_accu_{}.npy'.format(args.person_id), x)

# total = 0
# for i in range(10):
#     # if i==7:
#     #     continue
#     x = np.load('./best_accu_{}.npy'.format(i))
#     x1 = np.load('./baseline/best_accu_{}.npy'.format(i))
#     print(x-x1)
#     total += x[0]
# print(total / 10)
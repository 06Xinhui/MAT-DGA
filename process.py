import os
import scipy.io as scio
import numpy as np
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--person_id', type=int, default=0, help='person index')
args=parser.parse_args()

# data = scio.loadmat('./data/sub10action7.mat')
# data = data['subject'][0]
data = scio.loadmat('./data/subject10action7_new.mat')
data = data['sub'][0]

window_size = 1
tgt_dom_id = [args.person_id]
num_persons = 10
num_actions = 7
shutil.rmtree('./data/source')
shutil.rmtree('./data/target')

os.makedirs('./data/source')
os.makedirs('./data/target')
for i in range(num_actions):
    os.mkdir('./data/source/'+str(i))
    os.mkdir('./data/target/'+str(i))
# images = data[0][0]
# for i in range(1, num_persons):
#     images = np.concatenate((images, data[i][0]), axis=0)

# mean = np.mean(images)
# std = np.std(images)
remove_list = [ ]
for i in range(num_persons):
    if i in remove_list:
        continue
    if i in tgt_dom_id:
        save_dir = './data/target'
        
    else:
        save_dir = './data/source'
    for j in range(num_actions):
        # images, labels = data[i][0][j]
        images, labels = data[i][0], data[i][1]
        mean = np.mean(images, axis=(0,1,2), keepdims=True)
        std = np.std(images, axis=(0, 1,2), keepdims=True)
        images = (images-mean) / (std + 1e-5)
        sample_num = images.shape[0] // window_size
        for k in range(sample_num):
            save_path= os.path.join(save_dir, str(labels[k][0]), str(i) + '_' + '%04d'%k + '.npy')
            np.save(save_path, images[k])
print('data processing finished!')
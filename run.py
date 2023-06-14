import os

def train(index):
    # command = 'cd ./data'
    # os.system(command)
    # import pdb
    # pdb.set_trace()
    command = 'python ./process.py --person_id=%d' % index
    os.system(command)
    command = 'CUDA_VISIBLE_DEVICES=6 python train.py --person_id=%d' % index
    os.system(command)
    command = 'python plot.py --person_id=%d' % index
    os.system(command)
train_list = list(range(0,10))
for i in train_list:
    train(index=i)





    
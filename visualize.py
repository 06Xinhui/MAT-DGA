import os
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from model import BaseNet
from dataset import MyDataset
import torch
import itertools

model = BaseNet(7)
model.load_state_dict(torch.load('./models/mixup_ad_adaptation_id2/best_2.pth'))
model.cuda().eval()


plt.figure()
source_img_feats = []
source_img_labels = []
source_img = os.listdir('./data/source')
source_img.sort()
for label in source_img:
    image_names = os.listdir(os.path.join('./data/source', label))
    image_names.sort()
    for name in image_names:
        path = os.path.join('./data/source', label, name)
        source_img_feats.append(np.load(path))
        source_img_labels.append(int(label))

source_img_feats = np.concatenate(source_img_feats, axis=0).reshape((-1, 16 * 8 * 5))

# source_img_feats = np.stack(source_img_feats, axis=0)
# source_img_feats = source_img_feats.transpose(0, 3, 1, 2)
# source_img_feats = torch.from_numpy(source_img_feats).float().cuda().contiguous()
# source_img_feats = model(source_img_feats).data.cpu().numpy()


target_img_feats = []
target_img_labels = []
target_img = os.listdir('./data/target')
target_img.sort()
for label in source_img:
    image_names = os.listdir(os.path.join('./data/target', label))
    image_names.sort()
    for name in image_names:
        path = os.path.join('./data/target', label, name)
        target_img_feats.append(np.load(path))
        target_img_labels.append(int(label))
target_img_feats = np.concatenate(target_img_feats, axis=0).reshape((-1, 16 * 8 * 5))

# target_img_feats = np.stack(target_img_feats, axis=0)
# target_img_feats = target_img_feats.transpose(0, 3, 1, 2)
# target_img_feats = torch.from_numpy(target_img_feats).float().cuda().contiguous()
# target_img_feats = model(target_img_feats).data.cpu().numpy()



img_feats = np.concatenate((source_img_feats, target_img_feats), axis=0)
leg = [0, 0, 0, 0, 0, 0, 0]
tsne = TSNE().fit_transform(img_feats)
source_tsne = tsne[:source_img_feats.shape[0]]
target_tsne = tsne[source_img_feats.shape[0]:]


source_points = [[] for i in range(7)]
for i in range(len(source_img_labels)):
    source_points[source_img_labels[i]].append(source_tsne[i])
for i in range(7):
    source_points[i] = np.stack(source_points[i], axis=0)
    plt.scatter(source_points[i][:,0],source_points[i][:,1],c='C'+str(i),s=5.0,marker='o', alpha=0.05)


# plt.xticks([])
# plt.yticks([])
# # plt.xlim((-80,80))
# # plt.ylim((-80,80))
# plt.legend(leg, ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'], loc=2, bbox_to_anchor=(0.855, 0.995), borderaxespad=0., fontsize=12)
# plt.tight_layout()
# plt.savefig('./figs/mixup_ad_generalization_source_2.png')
# plt.figure()

target_points = [[] for i in range(7)]
for i in range(len(target_img_labels)):
    target_points[target_img_labels[i]].append(target_tsne[i])
for i in range(len(target_points)):
    target_points[i] = np.stack(target_points[i], axis=0)
    leg[i]=plt.scatter(target_points[i][:,0],target_points[i][:,1],c='C'+str(i),s=5.0,marker='o')
# import pdb
# pdb.set_trace()
plt.xticks([])
plt.yticks([])
# plt.xlim((-80,80))
# plt.ylim((-80,80))
plt.legend(leg, ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7'], loc=2, bbox_to_anchor=(0.855, 0.995), borderaxespad=0., fontsize=12)
plt.tight_layout()
plt.savefig('./figs/mixup_ad_initial_target_2.png')





# def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
#     """
#     - cm : 计算出的混淆矩阵的值
#     - classes : 混淆矩阵中每一行每一列对应的列
#     - normalize : True:显示百分比, False:显示个数
#     """
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("显示百分比：")
#         np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
#         print(cm)
#     else:
#         print('显示具体数字：')
#         print(cm)
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     plt.ylim(len(classes) - 0.5, -0.5)
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], '.2f'),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.savefig('./figs/confusionmatrix_generalization_2.png')

# label_id = list(str(i) for i in range(7))
# target_img_feats = []
# target_img_labels = []
# target_img = os.listdir('./data/target')
# for label in label_id:
#     image_names = os.listdir(os.path.join('./data/target', label))
#     for name in image_names:
#         path = os.path.join('./data/target', label, name)
#         target_img_feats.append(np.load(path))
#         target_img_labels.append(int(label))
# # target_img_feats = np.concatenate(target_img_feats, axis=0).reshape((-1, 16 * 8 * 5))

# target_img_feats = np.stack(target_img_feats, axis=0)
# target_img_feats = target_img_feats.transpose(0, 3, 1, 2)
# target_img_feats = torch.from_numpy(target_img_feats).float().cuda().contiguous()
# target_img_feats = model(target_img_feats).data.cpu().numpy()
# x = np.zeros([7,7])
# for i in range(target_img_feats.shape[0]):
#     pred_label = np.argmax(target_img_feats[i])
#     x[target_img_labels[i],  pred_label] += 1
# print(x * 7 / target_img_feats.shape[0])

# plt.figure()
# names = ['G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7']
# plot_confusion_matrix(x, names)
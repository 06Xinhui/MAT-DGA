import torch
import torch.nn as nn
import numpy as np
from model import BaseNet
from dataset import MyDataset
import copy
import os
import random
import torch.nn.functional as F
from torch.distributions import Beta
import copy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--person_id', type=int, default=0, help='person index')
args=parser.parse_args()



def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True




def train(model, train_dataloader, test_dataloader, optimizer, scheduler, epoches):
    criterion = nn.CrossEntropyLoss()
    best_accu = 0
    iters = 0
    
    for epoch in range(epoches):
        
        print('training for epoch:{}'.format(epoch))
        model.train()
        for data in train_dataloader:
            imgs, labels, ids = data['img'].cuda(), data['label'].cuda(), data['id'].cuda()
            imgs, labels, ids = imgs.squeeze(0), labels.view(-1), ids.view(-1)

            imgs_mix, targets_a, targets_b, ids_a, ids_b, lam = mixup_data(imgs, labels, ids)

            out, out_d = model(imgs_mix)
            loss_c = mixup_criterion(criterion, out, targets_a, targets_b, lam)
            # optimizer.zero_grad()
            # loss.backward()
            # #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            # optimizer.step()

            # optimizer.zero_grad()
            out1, out2 = model(imgs)
            loss_d = criterion(out2, ids)
           

            loss = loss_c + loss_d
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

            iters += 1
            if iters % 10 == 0:
                #print('epoches:{}, iter:{}, loss:{:.4f}'.format(epoch, iters, loss.item()))
                print('epoches:{}, iter:{}, loss:{:.4f}, loss_d:{:.4f}'.format(epoch, iters, loss.item(), loss_d.item()))
        torch.save(model.state_dict(), os.path.join('./models/mixup_ad_adaptation', str(epoch+1) + '.pth'))
        # if epoch == 2:
        #     accumulate_accuracy(model, train_dataloader, test_dataloader, copy.deepcopy(test_dataloader))
        #     break
        test_model = adaptation(copy.deepcopy(model), train_dataloader, test_dataloader)
        accuracy = validate(test_model, test_dataloader)
        if accuracy > best_accu:
            torch.save(test_model.state_dict(), os.path.join('./models/mixup_ad_adaptation', 'best_'+str(args.person_id)+'.pth'))
            best_accu = accuracy
            np.save('./best_accu.npy', np.array([best_accu]))
        print('epoch {}, best accuracy is {:.4f}, current accuracy is {:.4f}'.format(epoch, best_accu, accuracy))
        epoch += 1
        scheduler.step()


def mixup_data(x, y, ids, alpha=0.5):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    # import pdb
    # pdb.set_trace()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    ids_a, ids_b = ids, ids[index]
    return mixed_x, y_a, y_b, ids_a, ids_b, lam

def mixup_test_data(train_imgs, test_imgs, alpha=0.5):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    mixed_x = lam * train_imgs + (1 - lam) * test_imgs
    return mixed_x, lam    


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def accumulate_accuracy(model, labeled_loader, unlabeled_loader, test_loader):
    model.train()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,  weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    infer_model = copy.deepcopy(model).eval()
    train_iterloader = iter(labeled_loader)
    accu = []

    hist_imgs = []
    hist_labels = []
    for data in unlabeled_loader:
        test_imgs  = data['img'].cuda()
        test_labels = torch.max(infer_model(test_imgs), dim=1)[1]
        hist_imgs.append(test_imgs)
        hist_labels.append(test_labels)
        new_model = copy.deepcopy(model).train()
        optimizer = torch.optim.Adam(new_model.parameters(), lr=1e-4,  weight_decay=1e-3)
        accuracy = 0
        for _ in range(5):
            c = list(zip(hist_imgs, hist_labels))
            random.shuffle(c)
            hist_imgs, hist_labels = zip(*c)
            hist_imgs, hist_labels = list(hist_imgs), list(hist_labels)
            new_model.train()
            for test_imgs, test_labels in zip(hist_imgs, hist_labels):
                try:
                    train_data = next(train_iterloader)
                except:
                    train_iterloader = iter(labeled_loader)
                    train_data = next(train_iterloader)
                train_imgs, train_labels = train_data['img'].cuda(), train_data['label'].cuda()
                imgs_mix, lam = mixup_test_data(train_imgs, test_imgs)
                out1, out2 = new_model(imgs_mix)
                loss_c = mixup_criterion(criterion, out1, train_labels, test_labels, lam)

                ad_data = torch.cat((train_imgs, test_imgs), dim=0)
                _, ad_out = new_model(ad_data, mode='adaptation')
                ad_labels = torch.cat((torch.ones_like(train_labels), torch.zeros_like(train_labels)), dim=0)
                loss_d = criterion(ad_out, ad_labels)

                loss = loss_c + loss_d
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(new_model.parameters(), 10)
                optimizer.step()
            accuracy_tmp = validate(new_model.eval(), test_loader)
            accuracy = max(accuracy_tmp, accuracy)
        accu.append(accuracy)
    # import time
    # t1 = time.time()
    # for data in unlabeled_loader:
    #     model.train()
    #     test_imgs  = data['img'].cuda()
    #     test_labels = torch.max(infer_model(test_imgs), dim=1)[1]
    #     # try:
    #     #     train_data = next(train_iterloader)
    #     # except:
    #     #     train_iterloader = iter(labeled_loader)
    #     #     train_data = next(train_iterloader)
    #     # train_imgs, train_labels = train_data['img'].cuda(), train_data['label'].cuda()
    #     # imgs_mix, lam = mixup_test_data(train_imgs, test_imgs)
    #     for _ in range(10):
    #         try:
    #             train_data = next(train_iterloader)
    #         except:
    #             train_iterloader = iter(labeled_loader)
    #             train_data = next(train_iterloader)
    #         train_imgs, train_labels = train_data['img'].cuda(), train_data['label'].cuda()
    #         # imgs_mix, lam = mixup_test_data(train_imgs, test_imgs)
    #         # out1, out2 = model(imgs_mix)
    #         # loss_c = mixup_criterion(criterion, out1, train_labels, test_labels, lam)
    #         out1, out2 = model(train_imgs)
    #         loss_c = criterion(out1, train_labels)

    #         ad_data = torch.cat((train_imgs, test_imgs), dim=0)
    #         _, ad_out = model(ad_data, mode='adaptation')
    #         ad_labels = torch.cat((torch.ones_like(train_labels), torch.zeros_like(train_labels)), dim=0)
    #         loss_d = criterion(ad_out, ad_labels)

    #         loss = loss_c + loss_d
    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    #         optimizer.step()
    #     accuracy = validate(model.eval(), test_loader)
    #     accu.append(accuracy)
    np.save('accu_' + str(args.person_id) + '.npy', accu)

def adaptation(model, train_loader, test_loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,  weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    infer_model = copy.deepcopy(model).eval()
    train_iterloader = iter(train_loader)
    for _ in range(10):
        for data in test_loader:
            test_imgs  = data['img'].cuda()
            test_labels = torch.max(infer_model(test_imgs), dim=1)[1]
            try:
                train_data = next(train_iterloader)
            except:
                train_iterloader = iter(train_loader)
                train_data = next(train_iterloader)
            train_imgs, train_labels = train_data['img'].cuda(), train_data['label'].cuda()
            imgs_mix, lam = mixup_test_data(train_imgs, test_imgs)
            out1, out2 = model(imgs_mix)
            #out1,_ = model(test_imgs)
            #loss_c = criterion(out1, test_labels)
            loss_c = mixup_criterion(criterion, out1, train_labels, test_labels, lam)

            ad_data = torch.cat((train_imgs, test_imgs), dim=0)
            _, ad_out = model(ad_data, mode='adaptation')
            ad_labels = torch.cat((torch.ones_like(train_labels), torch.zeros_like(train_labels)), dim=0)
            loss_d = criterion(ad_out, ad_labels)

            loss = loss_d + loss_c
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            #print('ADAPTATION, loss:{:.4f}, loss_c:{:.4f}, loss_d:{:.4f}'.format(loss.item(), loss_c.item(), loss_d.item()))
    return model.eval()

def validate(model, dataloader):
    model.eval()
    sample_num = 0.0
    correct = 0.0
    for _, data in enumerate(dataloader):
        image, label = data['img'].cuda(), data['label'].cuda()
        image, label = image.squeeze(0), label.view(-1)
        out = model(image)
        pred = torch.max(out, dim=1)[1]
        correct += torch.sum(pred==label).item()
        sample_num += image.size(0)
    return correct / sample_num


if __name__ == '__main__':
    setup_seed(0)
    class_num = 7
    lr = 1e-3
    epoches = 30
    batch_size = 50
    model = BaseNet(class_num)
    model.cuda()
    train_dataset = MyDataset('./data/source',subset='train')
    test_dataset = MyDataset('./data/target',subset='test')

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=10, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20])
    train(model, train_dataloader, test_dataloader, optimizer, scheduler, epoches)



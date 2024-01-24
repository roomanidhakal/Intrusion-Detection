import datetime
import os

import numpy as np
import torch
from torch.autograd import Variable

import util.image_tools as image_tools
from util.image_reader import TrainImageReader
from util.imagedb import ImageDB
from util.models import PNet, LossFn

root = 'C:/Users/hp/Documents/Thesis/MTCNN/'
annotation_file = root + 'dataset/annotation/pnet_image_annotation.txt'
model_store_path = root + 'model'
end_epoch = 10
frequent = 100
lr = 0.01
lr_epoch_decay = [9]
batch_size = 2
use_cuda = True


def compute_accuracy(prob_cls, gt_cls):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    # we only need the detection which >= 0
    mask = torch.ge(gt_cls, 0)
    # get valid element
    valid_gt_cls = torch.masked_select(gt_cls, mask)
    valid_prob_cls = torch.masked_select(prob_cls, mask)
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls, 0.6).float()
    right_ones = torch.eq(prob_ones, valid_gt_cls).float()

    # if size == 0 meaning that your gt_labels are all negative, landmark or part
    return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))


def train_pnet(imdb):
    # create lr_list
    lr_epoch_decay.append(end_epoch + 1)
    lr_list = np.zeros(end_epoch)
    lr_t = lr
    for i in range(len(lr_epoch_decay)):
        if i == 0:
            lr_list[0:lr_epoch_decay[i] - 1] = lr_t
        else:
            lr_list[lr_epoch_decay[i - 1] - 1:lr_epoch_decay[i] - 1] = lr_t
        lr_t *= 0.1

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    lossfn = LossFn()
    net = PNet(is_train=True, use_cuda=use_cuda)
    net.train()

    if use_cuda:
        net.cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr_list[0])
    print('model loaded', torch)
    train_data = TrainImageReader(imdb, 12, batch_size, shuffle=True)

    for cur_epoch in range(1, end_epoch + 1):
        train_data.reset()  # shuffle
        for param in optimizer.param_groups:
            param['lr'] = lr_list[cur_epoch - 1]
        for batch_idx, (image, (gt_label, gt_bbox, gt_landmark)) in enumerate(train_data):
            im_tensor = [image_tools.convert_image_to_tensor(image[i, :, :, :]) for i in range(image.shape[0])]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())
            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()

            cls_pred, box_offset_pred = net(im_tensor)
            cls_loss = lossfn.cls_loss(gt_label, cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)
            all_loss = cls_loss * 1.0 + box_offset_loss * 0.5

            if batch_idx % frequent == 0:
                accuracy = compute_accuracy(cls_pred, gt_label)

                show1 = accuracy.data.cpu().numpy()
                show2 = cls_loss.data.cpu().numpy()
                show3 = box_offset_loss.data.cpu().numpy()
                show5 = all_loss.data.cpu().numpy()

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s " % (
                    datetime.datetime.now(), cur_epoch, batch_idx, show1, show2, show3, show5, lr_list[cur_epoch - 1]))

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

    torch.save(net.state_dict(), os.path.join(model_store_path, "pnet_model.pt"))
    torch.save(net, os.path.join(model_store_path, "pnet_model.pkl"))


imagedb = ImageDB(annotation_file)
gt_imdb = imagedb.load_imdb()
print('DATASIZE', len(gt_imdb))
gt_imdb = imagedb.append_flipped_images(gt_imdb)
print('FLIP DATASIZE', len(gt_imdb))
train_pnet(gt_imdb)

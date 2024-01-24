import os
import time

import cv2
import numpy as np
from six.moves import cPickle

import util.vision as vision
from util import assemble
from util.detect import MtcnnDetector, create_mtcnn_net
from util.image_reader import TestImageLoader
from util.imagedb import ImageDB
from util.utils import convert_to_square, IoU

root = 'C:/Users/hp/Documents/Thesis/MTCNN/'
prefix_path = root + "dataset/widerface/WIDER_train/images/"
traindata_store = root + "dataset/train/"
pnet_model_file = root + 'model/pnet_model.pt'
annotation_file = root + "dataset/annotation/train_annotation.txt"
use_cuda = False
save_path = root + 'model/'
rnet_postive_file = root + 'dataset/annotation/pos_rnet.txt'
rnet_neg_file = root + 'dataset/annotation/neg_rnet.txt'
rnet_part_file = root + 'dataset/annotation/part_rnet.txt'
imglist_filename = root + 'dataset/annotation/rnet_image_annotation.txt'


def gen_rnet_data(data_dir, anno_file, pnet_model_file, prefix_path='', use_cuda=True, vis=False):
    pnet, _, _ = create_mtcnn_net(p_model_path=pnet_model_file, use_cuda=use_cuda)
    mtcnn_detector = MtcnnDetector(pnet=pnet, min_face_size=12)
    imagedb = ImageDB(anno_file, mode="test", prefix_path=prefix_path)
    imdb = imagedb.load_imdb()
    image_reader = TestImageLoader(imdb, 1, False)

    all_boxes = list()
    batch_idx = 0

    print('size:%d' % image_reader.size)
    for databatch in image_reader:
        if batch_idx % 100 == 0:
            print("%d images done" % batch_idx)
        im = databatch
        t = time.time()

        boxes, boxes_align = mtcnn_detector.detect_pnet(im=im)
        if boxes_align is None:
            all_boxes.append(np.array([]))
            batch_idx += 1
            continue
        if vis:
            rgb_im = cv2.cvtColor(np.asarray(im), cv2.COLOR_BGR2RGB)
            vision.vis_two(rgb_im, boxes, boxes_align)

        t1 = time.time() - t
        print('cost time ', t1)
        t = time.time()
        all_boxes.append(boxes_align)
        batch_idx += 1

    save_file = os.path.join(save_path, "detections_%d.pkl" % int(time.time()))
    with open(save_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    gen_rnet_sample_data(data_dir, anno_file, save_file, prefix_path)


def gen_rnet_sample_data(data_dir, anno_file, det_boxs_file, prefix_path):
    neg_save_dir = os.path.join(data_dir, "rnet/negative")
    pos_save_dir = os.path.join(data_dir, "rnet/positive")
    part_save_dir = os.path.join(data_dir, "rnet/part")

    for dir_path in [neg_save_dir, pos_save_dir, part_save_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    with open(anno_file, 'r') as f:
        annotations = f.readlines()

    image_size = 24
    im_idx_list = list()
    gt_boxes_list = list()
    num_of_images = len(annotations)
    print("processing %d images in total" % num_of_images)

    for annotation in annotations:
        annotation = annotation.strip().split(' ')
        im_idx = os.path.join(prefix_path, annotation[0])

        boxes = list(map(float, annotation[1:]))
        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        im_idx_list.append(im_idx)
        gt_boxes_list.append(boxes)

    f1 = open(os.path.join(rnet_postive_file), 'w')
    f2 = open(os.path.join(rnet_neg_file), 'w')
    f3 = open(os.path.join(rnet_part_file), 'w')
    det_handle = open(det_boxs_file, 'rb')
    det_boxes = cPickle.load(det_handle)
    print(len(det_boxes), num_of_images)
    n_idx = 0
    p_idx = 0
    d_idx = 0
    image_done = 0
    for im_idx, dets, gts in zip(im_idx_list, det_boxes, gt_boxes_list):

        gts = np.array(gts, dtype=np.float32).reshape(-1, 4)
        if gts.shape[0] == 0:
            continue
        if image_done % 100 == 0:
            print("%d images done" % image_done)
        image_done += 1

        if dets.shape[0] == 0:
            continue
        img = cv2.imread(im_idx)
        # change to square
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        neg_num = 0
        for box in dets:
            x_left, y_top, x_right, y_bottom, _ = box.astype(int)
            width = x_right - x_left + 1
            height = y_bottom - y_top + 1

            if width < 20 or x_left < 0 or y_top < 0 or x_right > img.shape[1] - 1 or y_bottom > img.shape[0] - 1:
                continue

            Iou = IoU(box, gts)
            cropped_im = img[y_top:y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3 and neg_num < 60:
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
                neg_num += 1
            else:
                idx = np.argmax(Iou)
                assigned_gt = gts[idx]
                x1, y1, x2, y2 = assigned_gt

                offset_x1 = (x1 - x_left) / float(width)
                offset_y1 = (y1 - y_top) / float(height)
                offset_x2 = (x2 - x_right) / float(width)
                offset_y2 = (y2 - y_bottom) / float(height)

                if np.max(Iou) >= 0.65:
                    save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                    f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1

                elif np.max(Iou) >= 0.4:
                    save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                    f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (
                        offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    f1.close()
    f2.close()
    f3.close()


gen_rnet_data(traindata_store, annotation_file, pnet_model_file, prefix_path, use_cuda)

anno_list = [rnet_postive_file, rnet_part_file, rnet_neg_file]

chose_count = assemble.assemble_data(imglist_filename, anno_list)
print("PNet train annotation result file path:%s" % imglist_filename)

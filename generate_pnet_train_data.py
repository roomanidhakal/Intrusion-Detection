import os

import cv2
import numpy as np

import util.assemble as assemble
from util.utils import IoU

root = 'C:/Users/hp/Documents/Thesis/MTCNN/'
anno_file = root + "dataset/annotation/train_annotation.txt"
im_dir = root + "dataset/widerface/WIDER_train/images/"
pos_save_dir = root + "dataset/train/pnet/positive"
part_save_dir = root + "dataset/train/pnet/part"
neg_save_dir = root + 'dataset/train/pnet/negative'
pnet_postive_file = root + 'dataset/annotation/pos_pnet.txt'
pnet_neg_file = root + 'dataset/annotation/neg_pnet.txt'
pnet_part_file = root + 'dataset/annotation/part_pnet.txt'
imglist_filename = root + 'dataset/annotation/pnet_image_annotation.txt'
path = root + 'dataset/widerface/wider_face_split/wider_face_train_bbx_gt.txt'

with open(path, 'r') as f:
    linelist = f.readlines()

out_f = open(anno_file, 'w')
ind = 0
Nlines = len(linelist)
while ind < Nlines:
    if linelist[ind][2] == '-':
        buf = linelist[ind][:-1]
        ind += 1
        N = int(linelist[ind])
        ind += 1
        for _ in range(N):
            bbox = list(map(int, linelist[ind].split()[:4]))
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            bbox = list(map(str, bbox))
            buf += (' ' + ' '.join(bbox))
            ind += 1
        out_f.write(buf + '\n')
    else:
        ind += 1
out_f.close()

f1 = open(pnet_postive_file, 'w')
f2 = open(pnet_neg_file, 'w')
f3 = open(pnet_part_file, 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
print("%d pics in total" % num)
p_idx = 0  # positive
n_idx = 0  # negative
d_idx = 0  # dont care
idx = 0
box_idx = 0
for annotation in annotations:
    annotation = annotation.strip().split(' ')
    im_path = os.path.join(im_dir, annotation[0])
    bbox = list(map(float, annotation[1:]))
    boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
    if boxes.shape[0] == 0:
        continue
    img = cv2.imread(im_path)
    idx += 1
    if idx % 100 == 0:
        print("%s images done, pos: %s part: %s neg: %s" % (idx, p_idx, d_idx, n_idx))

    height, width, channel = img.shape

    neg_num = 0
    while neg_num < 50:
        size = np.random.randint(12, min(width, height) / 2)
        nx = np.random.randint(0, width - size)
        ny = np.random.randint(0, height - size)
        crop_box = np.array([nx, ny, nx + size, ny + size])

        Iou = IoU(crop_box, boxes)

        cropped_im = img[ny: ny + size, nx: nx + size, :]
        resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

        if np.max(Iou) < 0.3:
            save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
            f2.write(save_file + ' 0\n')
            cv2.imwrite(save_file, resized_im)
            n_idx += 1
            neg_num += 1

    for box in boxes:
        x1, y1, x2, y2 = box
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        if max(w, h) < 40 or x1 < 0 or y1 < 0:
            continue
        if w < 12 or h < 12:
            continue
        
        for i in range(5):
            size = np.random.randint(12, min(width, height) / 2)
            delta_x = np.random.randint(max(-size, -x1), w)
            delta_y = np.random.randint(max(-size, -y1), h)
            nx1 = max(0, x1 + delta_x)
            ny1 = max(0, y1 + delta_y)

            if nx1 + size > width or ny1 + size > height:
                continue
            crop_box = np.array([nx1, ny1, nx1 + size, ny1 + size])
            Iou = IoU(crop_box, boxes)

            cropped_im = img[ny1: ny1 + size, nx1: nx1 + size, :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            if np.max(Iou) < 0.3:
                save_file = os.path.join(neg_save_dir, "%s.jpg" % n_idx)
                f2.write(save_file + ' 0\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1

        for i in range(20):
            size = np.random.randint(int(min(w, h) * 0.8), np.ceil(1.25 * max(w, h)))
            delta_x = np.random.randint(-w * 0.2, w * 0.2)
            delta_y = np.random.randint(-h * 0.2, h * 0.2)

            nx1 = max(x1 + w / 2 + delta_x - size / 2, 0)
            ny1 = max(y1 + h / 2 + delta_y - size / 2, 0)
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > width or ny2 > height:
                continue
            crop_box = np.array([nx1, ny1, nx2, ny2])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            cropped_im = img[int(ny1): int(ny2), int(nx1): int(nx2), :]
            resized_im = cv2.resize(cropped_im, (12, 12), interpolation=cv2.INTER_LINEAR)

            box_ = box.reshape(1, -1)
            if IoU(crop_box, box_) >= 0.65:
                save_file = os.path.join(pos_save_dir, "%s.jpg" % p_idx)
                f1.write(save_file + ' 1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                p_idx += 1
            elif IoU(crop_box, box_) >= 0.4:
                save_file = os.path.join(part_save_dir, "%s.jpg" % d_idx)
                f3.write(save_file + ' -1 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(save_file, resized_im)
                d_idx += 1
        box_idx += 1
f1.close()
f2.close()
f3.close()

anno_list = [pnet_postive_file, pnet_part_file, pnet_neg_file]

chose_count = assemble.assemble_data(imglist_filename, anno_list)
print("PNet train annotation result file path:%s" % imglist_filename)

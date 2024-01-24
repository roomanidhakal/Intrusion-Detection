import time

import cv2
import numpy as np
import torch
from torch.autograd.variable import Variable

import util.image_tools as image_tools
import util.utils as utils
from util.models import PNet, RNet, ONet


def create_mtcnn_net(p_model_path=None, r_model_path=None, o_model_path=None, use_cuda=True):
    pnet, rnet, onet = None, None, None

    if p_model_path is not None:
        pnet = PNet(use_cuda=use_cuda)
        if (use_cuda):
            print('p_model_path:{0}'.format(p_model_path))
            pnet.load_state_dict(torch.load(p_model_path))
            pnet.cuda()
        else:
            # forcing all GPU tensors to be in CPU while loading
            # pnet.load_state_dict(torch.load(p_model_path, map_location=lambda storage, loc: storage))
            pnet.load_state_dict(torch.load(p_model_path, map_location='cpu'))
        pnet.eval()

    if r_model_path is not None:
        rnet = RNet(use_cuda=use_cuda)
        if (use_cuda):
            print('r_model_path:{0}'.format(r_model_path))
            rnet.load_state_dict(torch.load(r_model_path))
            rnet.cuda()
        else:
            rnet.load_state_dict(torch.load(r_model_path, map_location=lambda storage, loc: storage))
        rnet.eval()

    if o_model_path is not None:
        onet = ONet(use_cuda=use_cuda)
        if (use_cuda):
            print('o_model_path:{0}'.format(o_model_path))
            onet.load_state_dict(torch.load(o_model_path))
            onet.cuda()
        else:
            onet.load_state_dict(torch.load(o_model_path, map_location=lambda storage, loc: storage))
        onet.eval()

    return pnet, rnet, onet


class MtcnnDetector(object):
    # P,R,O net face detection and landmarks align

    def __init__(self, pnet=None, rnet=None, onet=None, min_face_size=12, stride=2, threshold=[0.6, 0.7, 0.7],
                 scale_factor=0.709, ):
        self.pnet_detector = pnet
        self.rnet_detector = rnet
        self.onet_detector = onet
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor

    def unique_image_format(self, im):
        if not isinstance(im, np.ndarray):
            if im.mode == 'I':
                im = np.array(im, np.int32, copy=False)
            elif im.mode == 'I;16':
                im = np.array(im, np.int16, copy=False)
            else:
                im = np.asarray(im)
        return im

    def square_bbox(self, bbox):
        # convert bbox to square

        square_bbox = bbox.copy()

        h = bbox[:, 3] - bbox[:, 1] + 1
        w = bbox[:, 2] - bbox[:, 0] + 1
        l = np.maximum(h, w)
        square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
        square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5

        square_bbox[:, 2] = square_bbox[:, 0] + l - 1
        square_bbox[:, 3] = square_bbox[:, 1] + l - 1
        return square_bbox

    def generate_bounding_box(self, map, reg, scale, threshold):
        # generate bbox from feature map

        stride = 2
        cellsize = 12  # receptive field

        t_index = np.where(map[:, :, 0] > threshold)

        # find nothing
        if t_index[0].size == 0:
            return np.array([])

        # choose bounding box whose socre are larger than threshold
        dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]
        reg = np.array([dx1, dy1, dx2, dy2])

        score = map[t_index[0], t_index[1], 0]
        boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),  # x1 of prediction box in original image
                                 np.round((stride * t_index[0]) / scale),  # y1 of prediction box in original image
                                 np.round((stride * t_index[1] + cellsize) / scale),
                                 # x2 of prediction box in original image
                                 np.round((stride * t_index[0] + cellsize) / scale),
                                 # y2 of prediction box in original image
                                 # reconstruct the box in original image
                                 score,
                                 reg,
                                 # landmarks
                                 ])

        return boundingbox.T

    def resize_image(self, img, scale):

        # resize image and transform dimention to [batchsize, channel, height, width]
        height, width, channels = img.shape
        new_height = int(height * scale)  # resized new height
        new_width = int(width * scale)  # resized new width
        new_dim = (new_width, new_height)
        img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)  # resized image
        return img_resized

    def pad(self, bboxes, w, h):
        # pad the the boxes

        # width and height
        tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
        tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
        numbox = bboxes.shape[0]

        dx = np.zeros((numbox,))
        dy = np.zeros((numbox,))
        edx, edy = tmpw.copy() - 1, tmph.copy() - 1
        # x, y: start point of the bbox in original image
        # ex, ey: end point of the bbox in original image
        x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

        tmp_index = np.where(ex > w - 1)
        edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
        ex[tmp_index] = w - 1

        tmp_index = np.where(ey > h - 1)
        edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
        ey[tmp_index] = h - 1

        tmp_index = np.where(x < 0)
        dx[tmp_index] = 0 - x[tmp_index]
        x[tmp_index] = 0

        tmp_index = np.where(y < 0)
        dy[tmp_index] = 0 - y[tmp_index]
        y[tmp_index] = 0

        return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
        return_list = [item.astype(np.int32) for item in return_list]

        return return_list

    def detect_pnet(self, im):
        # Get face candidates through pnet

        # original wider face data
        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size  # find initial scale
        im_resized = self.resize_image(im, current_scale)  # scale = 1.0
        current_height, current_width, _ = im_resized.shape
        # fcn
        all_boxes = list()
        while min(current_height, current_width) > net_size:
            feed_imgs = []
            image_tensor = image_tools.convert_image_to_tensor(im_resized)
            feed_imgs.append(image_tensor)
            feed_imgs = torch.stack(feed_imgs)

            feed_imgs = Variable(feed_imgs)

            if self.pnet_detector.use_cuda:
                feed_imgs = feed_imgs.cuda()

            cls_map, reg = self.pnet_detector(feed_imgs)

            cls_map_np = image_tools.convert_chwTensor_to_hwcNumpy(cls_map.cpu())
            reg_np = image_tools.convert_chwTensor_to_hwcNumpy(reg.cpu())

            boxes = self.generate_bounding_box(cls_map_np[0, :, :], reg_np, current_scale, self.thresh[0])

            # generate pyramid images
            current_scale *= self.scale_factor  # self.scale_factor = 0.709
            im_resized = self.resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue

            # non-maximum suppresion
            keep = utils.nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None
        all_boxes = np.vstack(all_boxes)

        # merge the detection from first stage
        keep = utils.nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]

        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        boxes = np.vstack([all_boxes[:, 0],
                           all_boxes[:, 1],
                           all_boxes[:, 2],
                           all_boxes[:, 3],
                           all_boxes[:, 4],
                           ])

        boxes = boxes.T

        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        # refine the boxes
        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 all_boxes[:, 4],
                                 ])
        boxes_align = boxes_align.T

        # remove invalid box
        valindex = [True for _ in range(boxes_align.shape[0])]
        for i in range(boxes_align.shape[0]):
            if boxes_align[i][2] - boxes_align[i][0] <= 3 or boxes_align[i][3] - boxes_align[i][1] <= 3:
                valindex[i] = False
                print('pnet has one smaller than 3')
            else:
                if (boxes_align[i][2] < 1 or boxes_align[i][0] > w - 2 or boxes_align[i][3] < 1 or
                        boxes_align[i][1] > h - 2):
                    valindex[i] = False
                    print('pnet has one out')
        boxes_align = boxes_align[valindex, :]
        boxes = boxes[valindex, :]
        return boxes, boxes_align

    def detect_rnet(self, im, dets):
        # Get face candidates using rnet

        h, w, c = im.shape

        if dets is None:
            return None, None
        if dets.shape[0] == 0:
            return None, None

        # (705, 5) = [x1, y1, x2, y2, score, reg]
        # print("pnet detection {0}".format(dets.shape))
        # time.sleep(5)
        detss = dets
        # return square boxes
        dets = self.square_bbox(dets)
        detsss = dets
        # rounds
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            try:
                tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            except:
                print(dy[i], edy[i], dx[i], edx[i], y[i], ey[i], x[i], ex[i], tmpw[i], tmph[i])
                print(dets[i])
                print(detss[i])
                print(detsss[i])
                print(h, w)
                exit()
            crop_im = cv2.resize(tmp, (24, 24))
            crop_im_tensor = image_tools.convert_image_to_tensor(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg = self.rnet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None
        keep = utils.nms(boxes, 0.7)

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        boxes = np.vstack([keep_boxes[:, 0],
                           keep_boxes[:, 1],
                           keep_boxes[:, 2],
                           keep_boxes[:, 3],
                           keep_cls[:, 0],
                           ])

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0],
                                 ])

        boxes = boxes.T
        boxes_align = boxes_align.T

        # remove invalid box
        valindex = [True for _ in range(boxes_align.shape[0])]
        for i in range(boxes_align.shape[0]):
            if boxes_align[i][2] - boxes_align[i][0] <= 3 or boxes_align[i][3] - boxes_align[i][1] <= 3:
                valindex[i] = False
                print('rnet has one smaller than 3')
            else:
                if (boxes_align[i][2] < 1 or boxes_align[i][0] > w - 2 or boxes_align[i][3] < 1 or
                        boxes_align[i][1] > h - 2):
                    valindex[i] = False
                    print('rnet has one out')
        boxes_align = boxes_align[valindex, :]
        boxes = boxes[valindex, :]

        return boxes, boxes_align

    def detect_onet(self, im, dets):
        # Get face candidates using onet

        h, w, c = im.shape

        if dets is None:
            return None, None
        if dets.shape[0] == 0:
            return None, None

        detss = dets
        dets = self.square_bbox(dets)

        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = self.pad(dets, w, h)
        num_boxes = dets.shape[0]

        # cropped_ims_tensors = np.zeros((num_boxes, 3, 24, 24), dtype=np.float32)
        cropped_ims_tensors = []
        for i in range(num_boxes):
            try:
                tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
                # crop input image
                tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            except:
                print(dy[i], edy[i], dx[i], edx[i], y[i], ey[i], x[i], ex[i], tmpw[i], tmph[i])
                print(dets[i])
                print(detss[i])
                print(h, w)
            crop_im = cv2.resize(tmp, (48, 48))
            crop_im_tensor = image_tools.convert_image_to_tensor(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        if self.rnet_detector.use_cuda:
            feed_imgs = feed_imgs.cuda()

        cls_map, reg, landmark = self.onet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.thresh[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        keep = utils.nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0],
                                 ])

        boxes_align = boxes_align.T

        landmark = np.vstack([
            align_landmark_topx + keep_landmark[:, 0] * bw,
            align_landmark_topy + keep_landmark[:, 1] * bh,
            align_landmark_topx + keep_landmark[:, 2] * bw,
            align_landmark_topy + keep_landmark[:, 3] * bh,
            align_landmark_topx + keep_landmark[:, 4] * bw,
            align_landmark_topy + keep_landmark[:, 5] * bh,
            align_landmark_topx + keep_landmark[:, 6] * bw,
            align_landmark_topy + keep_landmark[:, 7] * bh,
            align_landmark_topx + keep_landmark[:, 8] * bw,
            align_landmark_topy + keep_landmark[:, 9] * bh,
        ])

        landmark_align = landmark.T

        return boxes_align, landmark_align

    def detect_face(self, img):
        # Detect face over image

        boxes_align = np.array([])
        landmark_align = np.array([])

        t = time.time()

        # pnet
        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])

            t1 = time.time() - t
            t = time.time()

        # rnet
        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

            t2 = time.time() - t
            t = time.time()

        # onet
        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

            t3 = time.time() - t
            t = time.time()
            print(
                "time cost " + '{:.3f}'.format(t1 + t2 + t3) + '  pnet {:.3f}  rnet {:.3f}  onet {:.3f}'.format(t1, t2,
                                                                                                                t3))

        return boxes_align, landmark_align

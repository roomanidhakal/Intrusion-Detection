import os

import cv2

from util.detect import create_mtcnn_net, MtcnnDetector
from util.vision import vis_face_on_image

root = 'C:/Users/hp/Documents/Thesis/MTCNN/'
p_model_path = root + 'model/pnet_model.pt'
r_model_path = root + 'model/rnet_model.pt'
o_model_path = root + 'model/onet_model.pt'
image_folder = root + 'test_images/input/'
image_output_folder = root + 'test_images/output/'

pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model_path, r_model_path=r_model_path, o_model_path=o_model_path,
                                    use_cuda=True)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24, threshold=[0.6, 0.7, 0.7])

image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

for file in image_files:
    img_path = os.path.join(image_folder, file)
    img = cv2.imread(img_path)
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxs, landmarks = mtcnn_detector.detect_face(img)

    save_name = os.path.join(image_output_folder, 'processed_' + file)
    vis_face_on_image(img_bg, bboxs, landmarks, save_name)

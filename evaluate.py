import os
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from util.detect import create_mtcnn_net, MtcnnDetector  # Importing from your module

root = 'C:/Users/hp/Documents/Thesis/MTCNN/'
wider_test_base_path = root + 'dataset/widerface/WIDER_val/images'
annotation_file_path = root + 'dataset/widerface/wider_face_split/wider_face_val_bbx_gt.txt'

# Load MTCNN model
pnet, rnet, onet = create_mtcnn_net(
    p_model_path=root+'model/pnet_model.pt',
    r_model_path=root+'model/rnet_model.pt',
    o_model_path=root+'model/onet_model.pt',
    use_cuda=True
)
mtcnn_detector = MtcnnDetector(pnet, rnet, onet, min_face_size=24, threshold=[0.6, 0.7, 0.7])

def load_annotations():
    annotations = {}
    with open(annotation_file_path, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        image_path = lines[i].strip()
        num_faces = int(lines[i + 1].strip())
        boxes = []
        for j in range(num_faces):
            box_info = list(map(int, lines[i + 2 + j].split()[:4]))
            boxes.append(box_info)
        annotations[image_path] = boxes
        i += num_faces + 2

    return annotations

def evaluate_model():
    y_true, y_pred = [], []

    annotations = load_annotations()

    for image_path, boxes in annotations.items():
        full_image_path = os.path.join(wider_test_base_path, image_path)
        image = cv2.imread(full_image_path)
        detected_boxes, _ = mtcnn_detector.detect_face(image)  # Detect faces

        for box in boxes:
            detected = any([is_face_detected(detected_box, box) for detected_box in detected_boxes])
            y_true.append(1)  # Face present
            y_pred.append(1 if detected else 0)  # Face detected or not

        for detected_box in detected_boxes:
            if not any([is_face_detected(detected_box, box) for box in boxes]):
                y_true.append(0)  # No face present
                y_pred.append(1)  # Face incorrectly detected

    conf_matrix = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:\n', conf_matrix)
    
    return y_true, y_pred

def is_face_detected(detected_box, true_box, iou_threshold=0.5):
    # Unpack the coordinates of both boxes
    x1_det, y1_det, x2_det, y2_det = detected_box[:4]
    x1_true, y1_true, x2_true, y2_true = true_box

    # Compute the (x, y)-coordinates of the intersection rectangle
    xA = max(x1_det, x1_true)
    yA = max(y1_det, y1_true)
    xB = min(x2_det, x2_true)
    yB = min(y2_det, y2_true)

    # Compute the area of intersection rectangle
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both the prediction and true boxes
    det_box_area = (x2_det - x1_det + 1) * (y2_det - y1_det + 1)
    true_box_area = (x2_true - x1_true + 1) * (y2_true - y1_true + 1)

    # Compute the intersection over union
    iou = inter_area / float(det_box_area + true_box_area - inter_area)

    # Return True if overlap is above the threshold
    return iou >= iou_threshold

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



if __name__ == "__main__":
    y_true, y_pred = evaluate_model()  # Evaluate the model

    # Compute the confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:\n', conf_matrix)

    # Compute and print the classification report
    report = classification_report(y_true, y_pred, target_names=['No Face', 'Face'])
    print('Classification Report:\n', report)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=['No Face', 'Face'],
                          title='Confusion matrix')

    plt.show()
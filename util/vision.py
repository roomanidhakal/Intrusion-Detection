import cv2
from matplotlib.patches import Circle


def vis_two(im_array, dets1, dets2, thresh=0.9):
    # Visualize detection results before and after calibration
    import matplotlib.pyplot as plt

    plt.subplot(121)
    plt.imshow(im_array)

    for i in range(dets1.shape[0]):
        bbox = dets1[i, :4]
        landmarks = dets1[i, 5:]
        score = dets1[i, 4]
        if score > thresh:
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor='red', linewidth=0.7)
            plt.gca().add_patch(rect)
            landmarks = landmarks.reshape((5, 2))
            for j in range(5):
                plt.scatter(landmarks[j, 0], landmarks[j, 1], c='yellow', linewidths=0.1, marker='x', s=5)
    plt.subplot(122)
    plt.imshow(im_array)

    for i in range(dets2.shape[0]):
        bbox = dets2[i, :4]
        landmarks = dets1[i, 5:]
        score = dets2[i, 4]
        if score > thresh:
            rect = plt.Rectangle((bbox[0], bbox[1]),
                                 bbox[2] - bbox[0],
                                 bbox[3] - bbox[1], fill=False,
                                 edgecolor='red', linewidth=0.7)
            plt.gca().add_patch(rect)

            landmarks = landmarks.reshape((5, 2))
            for j in range(5):
                plt.scatter(landmarks[j, 0], landmarks[j, 1], c='yellow', linewidths=0.1, marker='x', s=5)
    plt.show()


def vis_face_on_image(im_array, dets, landmarks, save_name):
    # Visualize detection results before and after calibration
    import pylab

    pylab.imshow(im_array)

    for i in range(dets.shape[0]):
        bbox = dets[i, :4]

        rect = pylab.Rectangle((bbox[0], bbox[1]),
                               bbox[2] - bbox[0],
                               bbox[3] - bbox[1], fill=False,
                               edgecolor='yellow', linewidth=0.9)
        pylab.gca().add_patch(rect)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((5, 2))
            for j in range(5):
                cir1 = Circle(xy=(landmarks_one[j, 0], landmarks_one[j, 1]), radius=2, alpha=0.4, color="red")
                pylab.gca().add_patch(cir1)

        pylab.savefig(save_name)
        pylab.show()


def vis_face_on_video(frame, bboxs, landmarks):
    for i in range(bboxs.shape[0]):
        bbox = bboxs[i, :4]
        frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 255), 2)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            for j in range(5):
                frame = cv2.circle(frame, (int(landmarks[i, j]), int(landmarks[i, j + 5])), 2, (0, 0, 255), -1)
    return frame

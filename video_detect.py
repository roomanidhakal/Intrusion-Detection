import os
import torch
import cv2

from util.detect import create_mtcnn_net, MtcnnDetector
from util.vision import vis_face_on_video

# MTCNN Model paths
root = 'C:/Users/hp/Documents/Thesis/MTCNN/'
p_model_path = root + 'model/pnet_model.pt'
r_model_path = root + 'model/rnet_model.pt'
o_model_path = root + 'model/onet_model.pt'
# Directory containing videos
video_folder = root + 'test_videos/input/'
video_output_folder = root + 'test_videos/output/'
VIDEO_SIZE = (640, 480)

# Create MTCNN Net
pnet, rnet, onet = create_mtcnn_net(p_model_path=p_model_path, r_model_path=r_model_path, o_model_path=o_model_path,
                                    use_cuda=True)
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24, threshold=[0.6, 0.7, 0.7])

# List all video files in the folder
video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]

# Process each video
for file in video_files:
    video_path = os.path.join(video_folder, file)
    cap = cv2.VideoCapture(video_path)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # After opening the video capture
    fps = cap.get(cv2.CAP_PROP_FPS)

    # When setting up the VideoWriter
    out = cv2.VideoWriter(os.path.join(video_output_folder, 'processed_' + file), fourcc, fps,
                          (int(cap.get(3)), int(cap.get(4))))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        bboxs, landmarks = mtcnn_detector.detect_face(frame)

        # Draw bounding boxes and landmarks on the frame
        frame_processed = vis_face_on_video(frame, bboxs, landmarks)

        # Write the processed frame
        out.write(frame_processed)

    # Release everything when job is finished
    cap.release()
    out.release()

    torch.cuda.empty_cache()

cv2.destroyAllWindows()

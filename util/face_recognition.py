import os
import torch
import cv2
import numpy
import torch.nn.functional as F
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

def initialize_facenet():
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    return facenet


def generate_embedding(facenet, img_pil):
    preprocess = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img_tensor = preprocess(img_pil)
    embedding = facenet(img_tensor.unsqueeze(0)).detach()
    return embedding

def recognize_face(face_embedding, embedding_database, recognition_threshold):
    closest_person = None
    closest_distance = float('inf')

    for person_name, embeddings in embedding_database.items():
        for db_embedding in embeddings:
            face_embedding_tensor = face_embedding.clone().detach()
            db_embedding_tensor = db_embedding.clone().detach()
            dist = F.pairwise_distance(face_embedding_tensor, db_embedding_tensor).item()
            if dist < closest_distance:
                closest_distance = dist
                closest_person = person_name

    if closest_distance < recognition_threshold:
        return closest_person
    else:
        return None

def create_embedding_database(base_folder, mtcnn_detector, facenet):
    embedding_database = {}

    for person_folder in os.listdir(base_folder):
        person_path = os.path.join(base_folder, person_folder)
        
        if os.path.isdir(person_path):
            person_name = person_folder
            embeddings = []
            for filename in os.listdir(person_path):
                if filename.lower().endswith(('.jpg', '.png')):
                    img_path = os.path.join(person_path, filename)
                    img = Image.open(img_path)
                    img = img.convert('RGB')
                    img_cv = cv2.cvtColor(numpy.array(img), cv2.COLOR_RGB2BGR)
                    bboxs, landmarks = mtcnn_detector.detect_face(img_cv)

                    if bboxs is not None:
                        x1, y1, x2, y2 = [int(b) for b in bboxs[0, :4]]
                        face_crop = img_cv[y1:y2, x1:x2]
                        face_crop_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                        embedding = generate_embedding(facenet, face_crop_pil)
                        embeddings.append(embedding)

            if embeddings:
                embedding_database[person_name] = embeddings

    return embedding_database


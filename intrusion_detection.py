import os
import threading
import tkinter as tk
from tkinter import filedialog, Label

import cv2
from PIL import Image, ImageTk
from util.detect import create_mtcnn_net, MtcnnDetector
from util.face_recognition import initialize_facenet, create_embedding_database, recognize_face, generate_embedding


class FaceRecognitionApp:
    ROOT_DIRECTORY = 'C:/Users/hp/Documents/Thesis/MTCNN/'
    DATABASE_DIRECTORY = ROOT_DIRECTORY + 'database/'
    MODEL_PATHS = {
        'pnet': ROOT_DIRECTORY + 'model/pnet_model.pt',
        'rnet': ROOT_DIRECTORY + 'model/rnet_model.pt',
        'onet': ROOT_DIRECTORY + 'model/onet_model.pt'
    }
    MIN_FACE_SIZE = 24
    DETECTION_THRESHOLD = [0.6, 0.7, 0.7]
    RECOGNITION_THRESHOLD = 0.9
    FRAME_SKIP = 5
    WINDOW_WIDTH = 640

    def __init__(self):
        self.detected_intruder_frame = None
        self.detected_intruder_label = None
        self.images_frame = None
        self.root = None
        self.embedding_database = None
        self.facenet = None
        self.mtcnn_detector = None
        self.photo_images = []
        self.detected_persons = []
        self.photo_images_lock = threading.Lock()
        self.initialize_detector()
        self.initialize_gui()

    def initialize_detector(self):
        pnet, rnet, onet = create_mtcnn_net(
            p_model_path=self.MODEL_PATHS['pnet'],
            r_model_path=self.MODEL_PATHS['rnet'],
            o_model_path=self.MODEL_PATHS['onet'],
            use_cuda=True
        )
        self.mtcnn_detector = MtcnnDetector(pnet, rnet, onet, self.MIN_FACE_SIZE, self.DETECTION_THRESHOLD)
        self.facenet = initialize_facenet()
        self.embedding_database = create_embedding_database(self.DATABASE_DIRECTORY, self.mtcnn_detector, self.facenet)

    def initialize_gui(self):
        self.root = tk.Tk()
        self.root.title("Face Recognition")
        self.root.state('zoomed')
        intrusion_list_label = tk.Label(self.root, text="Intrusion List", font=("Helvetica", 18))
        intrusion_list_label.place(x=150, y=50)
        canvas = tk.Canvas(self.root, width=600)
        self.images_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=self.images_frame, anchor="nw", tags="self.images_frame")
        canvas.place(x=20, y=100, width=600, height=600)

        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        self.root.bind("<MouseWheel>", on_mousewheel)
        self.images_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        btn = tk.Button(self.root, text="Browse Video", command=self.select_video)
        btn.place(x=500, y=20)
        live_cam_button = tk.Button(self.root, text="Start Live Camera", command=self.start_live_camera)
        live_cam_button.place(x=650, y=20)
        self.detected_intruder_label = tk.Label(self.root, text="Detected Intruder", font=("Helvetica", 18))
        self.detected_intruder_label.place(x=1000, y=50)
        self.detected_intruder_frame = tk.Label(self.root)
        self.detected_intruder_frame.place(x=700, y=100, width=600, height=600)
        self.display_database_images()
        self.intruder_canvas = tk.Canvas(self.root, width=600)
        self.intruder_frame = tk.Frame(self.intruder_canvas)
        intruder_canvas_window = self.intruder_canvas.create_window((0, 0), window=self.intruder_frame, anchor="nw", tags="self.intruder_frame")
        self.intruder_canvas.place(x=700, y=100, width=600, height=600)
        self.intruder_canvas.bind("<Configure>", lambda e: self.intruder_canvas.configure(scrollregion=self.intruder_canvas.bbox("all")))
        self.intruder_frame.bind("<Configure>", self.on_intruder_frame_configure)
        self.photo_images_intruder = []

    def on_intruder_frame_configure(self, event):
        self.intruder_canvas.configure(scrollregion=self.intruder_canvas.bbox("all"))

    def reset_detected_persons(self):
        self.detected_persons.clear()
        for widget in self.intruder_frame.winfo_children():
            widget.destroy()
        self.intruder_canvas.configure(scrollregion=(0, 0, 0, 0))

    def display_database_images(self):
        row = 0
        for folder in os.listdir(self.DATABASE_DIRECTORY):
            folder_path = os.path.join(self.DATABASE_DIRECTORY, folder)
            if os.path.isdir(folder_path):
                for preferred_file in ['1.jpg', '1.png']:
                    file_path = os.path.join(folder_path, preferred_file)
                    if os.path.isfile(file_path):
                        self.display_image_with_name(preferred_file, row, folder_path)
                        row += 1
                        break

    def display_image_with_name(self, filename, index, folder_path):
        col = index % 3
        grid_row = index // 3
        image_path = os.path.join(folder_path, filename)
        image = Image.open(image_path)
        image.thumbnail((200, 200))
        photo = ImageTk.PhotoImage(image)
        label = Label(self.images_frame, image=photo)
        label.image = photo
        label.grid(row=grid_row * 2, column=col, padx=10, pady=10)
        folder_name = os.path.basename(folder_path)
        name_label = Label(self.images_frame, text=folder_name)
        name_label.grid(row=grid_row * 2 + 1, column=col, padx=10, pady=10)

    def select_video(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.reset_detected_persons()
            self.play_video(file_path)

    def play_video(self, file_path):
        def video_loop():
            cap = cv2.VideoCapture(file_path)
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.FRAME_SKIP == 0:
                    ratio = self.WINDOW_WIDTH / frame.shape[1]
                    desired_height = int(frame.shape[0] * ratio)
                    resized_frame = cv2.resize(frame, (self.WINDOW_WIDTH, desired_height))
                    frame_processed = self.process_frame(resized_frame)
                    window_name = 'Video'
                    cv2.imshow(window_name, frame_processed)
                    cv2.moveWindow(window_name, 15, 120)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_count += 1

            cap.release()
            cv2.destroyAllWindows()

        thread = threading.Thread(target=video_loop)
        thread.start()

    def start_live_camera(self):
        self.reset_detected_persons()

        def camera_loop():
            cap = cv2.VideoCapture(0)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                ratio = self.WINDOW_WIDTH / frame.shape[1]
                desired_height = int(frame.shape[0] * ratio)
                resized_frame = cv2.resize(frame, (self.WINDOW_WIDTH, desired_height))
                frame_processed = self.process_frame(resized_frame)
                window_name = 'Live Camera'
                cv2.imshow(window_name, frame_processed)
                cv2.moveWindow(window_name, 15, 120)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        thread = threading.Thread(target=camera_loop)
        thread.start()

    def process_frame(self, frame):
        bboxs, landmarks = self.mtcnn_detector.detect_face(frame)
        if bboxs is not None:
            for bbox in bboxs:
                frame, person_name = self.process_face(frame, bbox)
                if person_name and person_name not in self.detected_persons:
                    self.detected_persons.append(person_name)
                    self.display_detected_persons()
        return frame

    def process_face(self, frame, bbox):
        x1, y1, x2, y2 = [int(b) for b in bbox[:4]]
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x1 >= frame.shape[1] or y1 >= frame.shape[0] or x2 <= 0 or y2 <= 0 or x1 >= x2 or y1 >= y2:
            return frame, None
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return frame, None
        face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        face_embedding = generate_embedding(self.facenet, face_pil)
        person_name = recognize_face(face_embedding, self.embedding_database, self.RECOGNITION_THRESHOLD)

        color, text = ((0, 255, 255), None)
        if person_name:
            color, text = ((0, 0, 255), person_name)
            print(f"Alert: {person_name} recognized!")

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if text:
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return frame, person_name

    def display_detected_persons(self):
        for index, person_name in enumerate(self.detected_persons):
            person_folder_path = os.path.join(self.DATABASE_DIRECTORY, person_name)
            if os.path.isdir(person_folder_path):
                for file in os.listdir(person_folder_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filename = file
                        break
                else:
                    filename = None

                self.display_person_image_with_name(filename, index, person_name, person_folder_path)

    def display_person_image_with_name(self, filename, index, person_name, folder_path):
        col = index % 3
        grid_row = index // 3

        if filename:
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            image.thumbnail((200, 200))

            with self.photo_images_lock:
                photo = ImageTk.PhotoImage(image)
                self.photo_images_intruder.append(photo)

            label = Label(self.intruder_frame, image=photo)
            label.image = photo
            label.grid(row=grid_row * 2, column=col, padx=10, pady=10)
            name_label = Label(self.intruder_frame, text=person_name)
            name_label.grid(row=grid_row * 2 + 1, column=col, padx=10, pady=10)
        else:
            label = Label(self.intruder_frame, text="Image Not Found")
            label.grid(row=grid_row * 2, column=col, padx=10, pady=10)

if __name__ == "__main__":

    app = FaceRecognitionApp()
    app.root.mainloop()
		

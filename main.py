import cv2
import os
from mtcnn import MTCNN
from helpers import draw_detected_features, crop_face
from facenet_pytorch import InceptionResnetV1

dataset_path = "lfw\lfw-deepfunneled\lfw-deepfunneled"

folder_list = [folder for folder in os.listdir(dataset_path)]
face_detector = MTCNN()

for folder in folder_list[:1]:
    folder_path = os.path.join(dataset_path, folder)
    images_path = [os.path.join(folder_path, image) for image in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, image))]

    for image_path in images_path:
        image = cv2.imread(image_path)
        file_name = image_path.split("\\")[-1]
        detected_features = face_detector.detect_faces(image)

        cv2.imshow(file_name, image)
        cv2.waitKey(-1)

        draw_image = draw_detected_features(image, detected_features)
        cropped_face = crop_face(image, detected_features)

        cv2.imshow(file_name, cropped_face)
        cv2.waitKey(-1)



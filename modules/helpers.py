import cv2
import numpy as np
import os

# Draws a bounding box and detected features on the face
def draw_detected_features(image, features):
    display_image = image.copy()
    for feature in features:
        x, y, width, height = feature['box']
        cv2.rectangle(display_image, (x,y), (x+width, y+height), (0, 255, 0), 2)

        for keypoint in feature['keypoints'].values():
            cv2.circle(display_image, keypoint, 2, (255, 0, 0), 2)
        return display_image

# Crops the face based on the detected bounding box
def crop_face(image, features):
    cropped_image = image.copy()

    x,y, width, height = features['box']
    return cropped_image[x:x+width, y:y+height]

# Preprocesses the input image before it can be embedded
def preprocess(image, input_shape):
    w, h, dim = input_shape
    batch_size = 1

    image_copy = image.copy()

    resized_image = cv2.resize(image_copy, (w, h))
    resized_image = resized_image.astype('float32')
    mean, std = resized_image.mean(), resized_image.std()
    normalized_image = (resized_image - mean) / std
    preprocessed_image = np.reshape(normalized_image, (batch_size, w, h, dim))  

    return preprocessed_image

# Returns a dictionary for the test data 
# In this dict, keys = person's name and values = the corresponding image file in the testing set
def get_test_files(test_dataset_path):
    test_set= {}

    for filename in os.listdir(test_dataset_path):

        person = filename[:-9]  # removes the '_XXXX.jpg' at the end of each file
        test_set[person] = filename
    
    return test_set

# Uses MTCNN to detect the facial features and bounding box associated with the image
# Creates an embedding for the cropped face
def detect_and_embed(face_detector, inception_resnet, image, resnet_input_shape):

    detected_features = face_detector.detect_faces(image)

    if not detected_features:
        print("No faces detected in this image")
        return None
    
    cropped_face = crop_face(image, detected_features[0])
    preprocessed_image = preprocess(cropped_face, resnet_input_shape)
    embedding = inception_resnet.predict(preprocessed_image)

    return embedding



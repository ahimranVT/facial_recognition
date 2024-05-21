import cv2
import os
from mtcnn import MTCNN
import pickle
import tensorflow as tf
from modules.helpers import get_test_files, detect_and_embed
from sklearn.metrics.pairwise import cosine_similarity

resnet_input_shape = (160, 160, 3)
inception_resnet = tf.keras.applications.InceptionResNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=resnet_input_shape)

train_dataset_path = "lfw\lfw-deepfunneled\lfw-deepfunneled"
test_dataset_path = 'lfw\lfw-deepfunneled\lfw-test'
embeddings_filepath = 'embeddings\embeddings.pkl'


folder_list = [folder for folder in os.listdir(train_dataset_path)]
face_detector = MTCNN()
person_dict = {}

# If no embeddings file exists, create a pickle file with embeddings for all faces in the training dataset
if not os.path.isfile(embeddings_filepath):

    for folder_name in folder_list:
        folder_path = os.path.join(train_dataset_path, folder_name)
        images_path = [os.path.join(folder_path, image) for image in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, image))]
        
        personwise_embeddings = []

        for image_path in images_path:
            image = cv2.imread(image_path)

            file_name = image_path.split("\\")[-1]

            embedding = detect_and_embed(face_detector, inception_resnet, image, resnet_input_shape)
            personwise_embeddings.append(embedding)
        
        person_dict[folder_name] = personwise_embeddings

    with open(embeddings_filepath, 'wb') as file:
        pickle.dump(person_dict, file)

# If the embeddings have been previously created, load them instead
else:
    with open(embeddings_filepath, 'rb') as file:
            person_dict = pickle.load(file)

test_files = get_test_files(test_dataset_path)


for person_name, file_name in test_files.items():
    similarity_scores = {}

    filepath = test_dataset_path + "\\" + file_name
    test_image = cv2.imread(filepath)

    test_embedding = detect_and_embed(face_detector, inception_resnet, test_image, resnet_input_shape)

    if test_embedding is None:
         print(f"The image of {person_name}could not be properly detected")
         continue
    
    for folder_name, embedding_list in person_dict.items():
        score_list = []

        for embedding in embedding_list:
            if embedding is not None:
                score = cosine_similarity(test_embedding, embedding)
                score_list.append(score)

        if score_list:
            similarity_scores[folder_name] = max(score_list)
        else:
            similarity_scores[folder_name] = 0
    

    print(f"The desired truth value is {person_name}, and the similarity score for this person was {similarity_scores[person_name]}")

    best_similarity_score = dict(sorted(similarity_scores.items(), key=lambda item: item[1], reverse=True)[:1])
    score = best_similarity_score.values()

    print(f"The model output value is {best_similarity_score.keys()}, and the similarity score for this person was {score}")


        

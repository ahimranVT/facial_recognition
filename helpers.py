import cv2

def draw_detected_features(image, features):
    display_image = image.copy()
    for feature in features:
        x, y, width, height = feature['box']
        cv2.rectangle(display_image, (x,y), (x+width, y+height), (0, 255, 0), 2)

        for keypoint in feature['keypoints'].values():
            cv2.circle(display_image, keypoint, 2, (255, 0, 0), 2)
        return display_image

def crop_face(image, features):
    cropped_image = image.copy()

    for feature in features:
        x, y, width, height = feature['box']
        return cropped_image[y:y+height, x:x+width]

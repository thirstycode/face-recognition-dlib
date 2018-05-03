import cv2
import time
import face_recognition
import os
import numpy as np
import sys
from save import *
from config_knn import number_of_times_to_upsample
from config_knn import num_jitters
from config_knn import distance_threshold
import math
from sklearn import neighbors
import os.path
import pickle
from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder

name_status = {}
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    global name_status

    X = []
    y = []
    status = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        print("Training On " + class_dir)
        # Loop through each training image for the current person
        for img_path in os.listdir(os.path.join(train_dir, class_dir)):
            img_path = os.path.join(train_dir, class_dir) +"/" + img_path
            if img_path == os.path.join(train_dir, class_dir) + "/" + "status.txt" :
                with open(img_path,'r+') as name:
                    content = name.read()
                    content = content.lower()
                    status.append(content)

            else :
                print("- " + img_path)
                image = face_recognition.load_image_file(img_path)
                face_bounding_boxes = face_recognition.face_locations(image)

                if len(face_bounding_boxes) != 1:
                    # If there are no people (or too many people) in a training image, skip the image.
                    if verbose:
                        print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
                else:
                    # Add face encoding for current image to the training set
                    X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes,num_jitters=num_jitters)[0])
                    y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    new_names =[]
    for name in y:
        if not name in new_names :
            new_names.append(name)
        else:
            continue
    name_status = dict(zip(new_names, status))
    print(name_status)
    knn_clf.fit(X, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    return knn_clf

def predict(X_img, knn_clf=None, model_path=None, distance_threshold=0.6):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    X_face_locations = face_recognition.face_locations(X_img,number_of_times_to_upsample=number_of_times_to_upsample)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations,num_jitters=num_jitters)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]

def show_prediction_labels_on_image(img, predictions):
    global name_status

    if len(predictions) > 0 :
        for name, (top, right, bottom, left) in predictions:
            if name != "unknown":
                colour_2 = colour_f(name_status[name])
            else:
                colour_2 = colour_f("unknown")
            draw_rectangle2(img, (left,top,right,bottom),colour_2)
            draw_text(img,name , left, top-5)
    else :
        print("not found")

    return img

def colour_f(status1):
    if status1=="vip":
        return (0,255,0)
    if status1=="blacklisted":
        return (0,0,255)
    else :
        return (255,255,255)

def draw_rectangle2(img, rect,colour):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (w, h), colour, 2)

def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

def show_result(frame):
    global faces
    global status
    predictions = predict(frame, model_path="trained_knn_model.clf",distance_threshold=distance_threshold)
    for name, (top, right, bottom, left) in predictions:
        print("- Found {} at ({}, {})".format(name, left, top))

    # Display results overlaid on an image
    save(frame,"before")
    frame = show_prediction_labels_on_image(frame, predictions)
    save(frame,"after")
    cv2.imshow("face_detected",frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("Training KNN classifier...")
classifier = train("knn-train", model_save_path="trained_knn_model.clf", n_neighbors=2)
print("Training complete!")

vid = cv2.VideoCapture(0)
# vid.open('http://192.168.43.1:8080/video')

while True:
    check , frame = vid.read()
    cv2.imshow("Press c to Pass frame , q to exit",frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    if key == ord('c'):
        show_result(frame)


vid.release()
cv2.destroyAllWindows()

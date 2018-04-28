from PIL import Image
import face_recognition
import cv2
import os
import numpy as np
import sys
import time

unknown_image_path = "214.jpg"

subjects = []
status = []
face_encoding_list = []

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

def prepare_training_data(data_folder_path):
    global subjects
    global status

    dirs = os.listdir(data_folder_path)

    faces = []

    labels = []

    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue;

        label = int(dir_name.replace("s", ""))

        subject_dir_path = data_folder_path + "/" + dir_name

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue;

            if image_name == "name.txt":
                name_path = subject_dir_path + "/" + image_name
                with open(name_path,'r+') as name:
                    content = name.read()
                    content = content.lower()
                    subjects.append(content)

            elif image_name == "status.txt":
                name_path = subject_dir_path + "/" + image_name
                with open(name_path,'r+') as name:
                    content = name.read()
                    content = content.lower()
                    status.append(content)

            else :
                image_path = subject_dir_path + "/" + image_name
                image = face_recognition.load_image_file(image_path)

                # make sure to resize on fixed amount
                # cv2.imshow("Training on image...", cv2.resize(image, (400, 500)))
                print("Faces Scanned: ", len(faces) + 1)
                cv2.waitKey(100)

                faces.append(image)
                labels.append(label)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces

faces = prepare_training_data("train-images")

# Load the jpg file into a numpy array
# image = face_recognition.load_image_file("manushi.jpg")
# biden_image = face_recognition.load_image_file("biden.jpg")
# manushi_image = face_recognition.load_image_file("manushi1.jpg")
unknown_image = face_recognition.load_image_file(unknown_image_path)
# Find all the faces in the image using the default HOG-based model.
# This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.
# See also: find_faces_in_picture_cnn.py
face_locations = face_recognition.face_locations(unknown_image)

try:
    # biden_face_encoding = face_recognition.face_encodings(biden_image)[0]
    # manushi_face_encoding = face_recognition.face_encodings(manushi_image)[0]
    for face in faces:
        face_encoding = face_recognition.face_encodings(face)[0]
        face_encoding_list.append(face_encoding)
except IndexError:
    print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
    quit()

print("I found {} face(s) in this photograph.".format(len(face_locations)))

img = unknown_image.copy()

for face_location in face_locations:

    # Print the location of each face in this image
    top, right, bottom, left = face_location
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

    # You can access the actual face itself like this:
    face_image = unknown_image[top:bottom, left:right]
    face_encoding1 = face_recognition.face_encodings(face_image)
    if len(face_encoding1) > 0 :
        face_encoding = face_encoding1[0]
        results = face_recognition.compare_faces(face_encoding_list, face_encoding)
        if True in results:
            index1 = results.index(True)
            colour_2 = colour_f(status[index1])
            draw_rectangle2(img, (left,top,right,bottom),colour_2)
            draw_text(img,subjects[index1] , left, top-5)

            print(subjects[index1])
            print(status[index1])
        else :
            print("not found")
            draw_rectangle2(img, (left,top,right,bottom),(255,255,255))
            draw_text(img,"No Match" ,left, top-5)
    else:
        draw_rectangle2(img, (left,top,right,bottom),(255,255,255))
        draw_text(img,"Error 1" ,left, top-5)



cv2.imshow("face_detected",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

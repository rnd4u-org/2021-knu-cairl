import face_recognition
import os
from tqdm import tqdm
from PIL import Image

def get_faces(image):
    face_locations = face_recognition.face_locations(image)
    im = []
    for loc in face_locations:
        im.append(image[loc[0]:loc[2], loc[3]:loc[1]])
    return im

def compare(e1, e2):
    results = face_recognition.compare_faces([e1], e2)
    return results[0]

def generate_faces(imagesPath="./data/images", facesPath="./data/faces"):
    try:
        os.mkdir(facesPath)
    except Exception:
        pass

    i = 0
    print("generating faces")
    for im in tqdm(os.listdir(imagesPath)):
        image = face_recognition.load_image_file(os.path.join(imagesPath, im))
        faces = get_faces(image)
        for face in faces:
            jpg = Image.fromarray(face)
            jpg.save(os.path.join(facesPath,"face"+str(i)+".jpg"))
            i += 1

def load_faces(faces, facesPath="./data/faces"):
    print("loading faces")
    for f in tqdm(os.listdir(facesPath)):
        face = face_recognition.load_image_file(os.path.join(facesPath, f))
        faces.append([face, face_recognition.face_encodings(face)[0]])
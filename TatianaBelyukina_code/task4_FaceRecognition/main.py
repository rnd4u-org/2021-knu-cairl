import face_recognition

if __name__ == "__main__":
    image1 = face_recognition.load_image_file("image1.jpg")
    image2 = face_recognition.load_image_file("image2.jpg")

    enc1 = face_recognition.face_encodings(image1)
    enc2 = face_recognition.face_encodings(image2)

    for e1 in enc1:
        for e2 in enc2:
            print(face_recognition.compare_faces([e1], e2)[0])

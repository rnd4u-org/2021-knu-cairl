# Used this file omly to analyze data
# tried to see how accurate face_recognition
# would work on our dataset
# found out that it recognise people 
# without mask in 98% of cases and
# only 50-70% when person in with_mask
# so you if it cannot see any face thats 
# really likely to be a picture with mask
# however it works a lot worse then cnn
# trained to detect mask so this file mostly
# useles but still interesting research

import face_recognition
import datetime

start = datetime.datetime.now()


def getFaces(image):
    face_locations = face_recognition.face_locations(image, model='cnn')
    im = []
    for loc in face_locations:
        im.append(image[loc[0]:loc[2], loc[3]:loc[1]])
    return im


found = 0
lens = []
i = 1

while True:
    b = True
    try:
        image = face_recognition.load_image_file("./data/without_mask/without_mask_" + str(i) + ".jpg")
        b = False
        faces = getFaces(image)
        print(i, ":", len(faces))
        lens.append(len(faces))
        if len(faces) > 0:
            found += 1
        i += 1
    except Exception:
        if b:
            break
i -= 1

print(lens)
print()
print()
print("found faces on", found, "out of", i, "pictures")
print()
print("time duration:", (datetime.datetime.now() - start).seconds, "seconds")
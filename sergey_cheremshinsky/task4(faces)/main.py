# 1 Create folder ./data/image/
# 2 Put there your images
# Then code will get faces from 
# that data and split them
# into few dirs for each person
# that will be detected

import os
from PIL import Image
from load_faces import generate_faces, load_faces, compare

if __name__ == "__main__":
    faces = []
    people = []
    generate_faces()
    load_faces(faces)
    n = len(faces)
    connections = []

    for i in range(n):
        connections.append([])
        for j in range(n):
            connections[i].append(int(compare(faces[i][1], faces[j][1])))
        print(connections[i])

    people = []

    arr = [0] * n
    while 0 in arr:
        for i in range(n):
            if arr[i] == 0:
                for person in people:
                    pos = sum(connections[i][j] for j in person)
                    if (pos > len(person) // 2 and pos > 1) or (pos == 1 and len(person) == 1):
                        person.append(i)
                        arr[i] = 1
                        break
                if arr[i] == 0:
                    arr[i] = 1
                    people.append([i])

    for i in range(len(people)):
        try:
            os.mkdir("./data/person" + str(i))
        except Exception:
            pass
        for j in people[i]:
            jpg = Image.fromarray(faces[j][0])
            jpg.save("./data/person" + str(i) + "/face" + str(j) + ".jpg")

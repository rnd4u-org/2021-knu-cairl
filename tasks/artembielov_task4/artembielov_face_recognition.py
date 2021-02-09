from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import matplotlib.pyplot as plt

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=20, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(x):
    return x[0]


dataset = datasets.ImageFolder('D:/projects/Python/Mask/recognition/test_images')
dataset.idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

aligned = []
for x, y in loader:
    x_aligned = mtcnn(x)
    plt.imshow(x_aligned.permute(1, 2, 0))
    plt.show()
    if x_aligned is not None:
        aligned.append(x_aligned)

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()
threshold = 0.55
for e1 in embeddings:
    dists = [(e1 - e2).norm().item() for e2 in embeddings]
if dists[0] < threshold:
    print('Person on photos is the same person!')
else:
    print('Person on photos is not the same person!')

from LibraryDevice import *
from Dataset import *
from DataLoader import *

fig, axis = plt.subplots(3, 10, figsize=(15, 10))
images, labels = next(iter(train_loader))

for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        image, label = images[i], labels[i]

        ax.imshow(image.view(28, 28), cmap='binary')
        ax.set(title = f"{label}")

fig, axis = plt.subplots(3, 10, figsize=(15, 10))
images, labels = next(iter(valid_loader))

for i, ax in enumerate(axis.flat):
    with torch.no_grad():
        image, label = images[i], labels[i]

        ax.imshow(image.view(28, 28), cmap='binary')
        ax.set(title = f"{label}")
from LibraryDevice import *
from Dataset import *
from DataLoader import *
from PlotDataset import *
from Architecture import *

LEARNING_RATE = 0.001680

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

epochs = 1
valid_loss_min = np.Inf
train_losses, valid_losses = [], []
history_accuracy = []

for e in range(1, epochs + 1):
    running_loss = 0

    for images, labels in train_loader:
        if train_on_gpu:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()

        ps = model(images)

        loss = criterion(ps, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
    else:
        valid_loss = 0
        accuracy = 0

        with torch.no_grad():
            model.eval()
            for images, labels in valid_loader:
                if train_on_gpu:
                    images, labels = images.cuda(), labels.cuda()
                ps = model(images)

                _, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                valid_loss += criterion(ps, labels)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        model.train()

        train_losses.append(running_loss / len(train_loader))
        valid_losses.append(valid_loss / len(valid_loader))
        history_accuracy.append(accuracy / len(valid_loader))

        network_learned = valid_loss < valid_loss_min

        if e == 1 or e % 5 == 0 or network_learned:
            print(f"Epoch: {e}/{epochs}.. ",
                  f"Training Loss: {running_loss / len(train_loader):.3f}.. ",
                  f"Validation Loss: {valid_loss / len(valid_loader):.3f}.. ",
                  f"Test Accuracy: {accuracy / len(valid_loader):.3f}")

        if network_learned:
            valid_loss_min = valid_loss
            torch.save(model.state_dict(), 'model_mtl_mnist.pt')
            print('Detected network improvement, saving current model')

import matplotlib.pyplot as plt

plt.plot(train_losses, label='Training Loss')
plt.plot(valid_losses, label='Validation Loss')
plt.legend(frameon=False)

plt.plot(history_accuracy, label='Validation Accuracy')
plt.legend(frameon=False)


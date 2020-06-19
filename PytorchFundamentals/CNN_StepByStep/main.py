from LibraryDevice import *
from Dataset import *
from DataLoader import *
from PlotDataset import *
from Architecture import *
from Testing import *
from Evaluating import *


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

submissionset = DatasetSubmissionMNIST('test.csv', transform=transform)
submissionloader = torch.utils.data.DataLoader(submissionset, batch_size=BATCH_SIZE, shuffle=False)

submission = [['ImageId', 'Label']]

with torch.no_grad():
    model.eval()
    image_id = 1

    for images in submissionloader:
        if train_on_gpu:
            images = images.cuda()
        log_ps = model(images)
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)

        for prediction in top_class:
            submission.append([image_id, prediction.item()])
            image_id += 1

print(len(submission) - 1)

import csv

with open('submission.csv', 'w') as submissionFile:
    writer = csv.writer(submissionFile)
    writer.writerows(submission)

print('Submission Complete!')
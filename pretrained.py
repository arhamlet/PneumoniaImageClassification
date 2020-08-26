import torch
import torch.nn as nn
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms
import os
import pandas as pd
from skimage import io
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

learning_rate = 0.0001
resume=None
weight_decay = 0
momentum = 0.9
epochs = 25
batch_size = 32
log_interval = 240
number_workers = 0
create_validationset = True
seed = 1
save_model = True
init_padding=2
validation_size=0.2
random_seed=1
in_channels = 3
load_model = False
best_acc = 0
#   checking accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:           #for i, (images, labels) in enumerate(loader)
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        accuracy = float(num_correct)/float(num_samples)*100
        print(f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}")

        return accuracy


def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
    print("Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint):
    model.eval()
    print('Loading Checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint["epoch"]
    best_acc = checkpoint["acc"]
    print(f"=> loaded checkpoint at epoch {epoch})", checkpoint["epoch"])

class Xraysdataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)   #number of images

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)


#   Code execution starts here
print("Loading Dataset")
data_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,))     #transforms.Normalize(mean,std)
])


dataset = Xraysdataset(                       #generic_data
    csv_file="dataset_labels.csv",
    root_dir="chest_xray",
    transform=data_transforms,
)

train_set, valid_set, test_set = torch.utils.data.random_split(dataset, [4856, 300, 700])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=batch_size,shuffle= True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#   Tensorboard writer
writer = SummaryWriter(log_dir='graphs')
step = 0

print("Initializing Model")
#   model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(pretrained=True)
#for param in model.parameters():
#    param.requires_grad = False

model.classifier[6] = nn.Linear(4096,3)
print(model)
model.to(device)

# Visualize model in TensorBoard
images1, _ = next(iter(train_loader))
writer.add_graph(model, images1.to(device))
writer.close()


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#loading model
if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))

# time
start_time = time.time()

print("Start training")
#   Training the network
for epoch in range(epochs):
    epoch_start_time = time.time()
    losses = []
    total_batch_images = float(4856)
    batch_correct_pred = 0
    # save model
    # if batch_accuracy>best_acc:
    #     best_acc = batch_accuracy
    #     checkpoint = {'state_dict': model.state_dict(),'acc' : batch_accuracy, 'epoch' : epoch,  'optimizer': optimizer.state_dict()}
    #     save_checkpoint(checkpoint)

    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Get data to cuda if possible
        
        images = images.to(device=device)
        labels = labels.to(device=device)

        # forward
        scores = model(images)
        loss = criterion(scores, labels)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

        # visualizing Dataset images
        # img_grid = torchvision.utils.make_grid(images)
        # writer.add_image('Xray_images', img_grid)

        # calculation running accuracy
        model.eval()
        _, predictions = scores.max(1)
        num_correct = (predictions == labels).sum()
        batch_correct_pred += float(num_correct)

    print(batch_correct_pred)
    epoch_elapsed = (time.time() - epoch_start_time) / 60
    print(f'Epoch {epoch} completed in : {epoch_elapsed:.2f} min')

    batch_loss = sum(losses)/len(losses)
    batch_accuracy = (batch_correct_pred/total_batch_images)*100

    print(f"Cost at epoch {epoch} is {batch_loss}")
    print(f"Training accuracy at {epoch} is: {batch_accuracy:.2f}")
    # batch_accuracy = check_accuracy(train_loader, model)

    if batch_accuracy>best_acc:
        best_acc = batch_accuracy
        checkpoint = {'state_dict': model.state_dict(),'acc' : batch_accuracy, 'epoch' : epoch,  'optimizer': optimizer.state_dict()}
        save_checkpoint(checkpoint)

    writer.add_scalar('Training loss', batch_loss, global_step=step)
    writer.add_scalar('Training accuracy', batch_accuracy, global_step=step)
    step += 1


elapsed = (time.time() - start_time)/60
print(f'Training completed in: {elapsed:.2f} min')

# Accuracy Check

print("Checking accuracy on training Set")
check_accuracy(train_loader, model)

print("Checking accuracy on valid Set")
check_accuracy(valid_loader, model)

print("Checking accuracy on test Set")
check_accuracy(test_loader, model)

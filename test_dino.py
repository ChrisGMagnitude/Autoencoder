import torch
from vit_pytorch import ViT,Dino
from dataLoader import MagClassDataset
import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime

model_path = r'/mnt/field/test/ml/cg/DINO Models'
train_path = r'/root/docker_data/Autoencoder/hdf5/train.hdf5'
valid_path = r'/root/docker_data/Autoencoder/hdf5/valid.hdf5'
initial_weights = 'default'
current_time = datetime.now()
architecture = 'DINO ViT'
description = 'First attempt at DINO - ViT Autoencoding'
image_size = 416
batch_size=128#256
learning_rate = 3e-4
num_epochs = 2
max_batches = 3

train_dataset = MagClassDataset(train_path)
val_dataset = MagClassDataset(valid_path,augment=False)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True)#,num_workers=os.cpu_count())  
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=True)#,num_workers=os.cpu_count())  

log = {}
log['model_path'] = model_path
log['name'] = ' - '.join([description, str(current_time)[:-7].replace(':','')])
log['architecture'] = architecture
log['hdf5_file'] = train_path
log['initial_weights'] = initial_weights
log['crop_ranges'] = train_dataset.crop_ranges
log['crop_jitter'] = train_dataset.crop_jitter
log['max_white_noise'] = train_dataset.max_white_noise
log['batch_size'] = batch_size
log['lr'] = lr
log['num_epochs'] = num_epochs
log['max_batches'] = max_batches

os.mkdir(os.path.join(log['model_path'],log['name']))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")

model = ViT(
    image_size = image_size,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

learner = Dino(
    model,
    image_size = image_size,
    hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 256,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
)

opt = torch.optim.Adam(learner.parameters(), lr = learning_rate)

all_train_loss = []
all_val_loss = []

for epoch in range(num_epochs):
    learner.train()
    print('epoch =',epoch)
    count = 0
    running_train_loss = 0
    for i,images in tqdm.tqdm(enumerate(train_data_loader)):
        images = images[0]
        count += images.shape[0]
        
        loss = learner(images)
        running_train_loss += loss.item() * images.shape[0]
    
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('train loss',loss.item())
        if i==max_batches:
            break
    learner.update_moving_average() # update moving average of teacher encoder and teacher centers
    epoch_train_loss = running_train_loss/(count)
    print('epoch_train_loss',epoch_train_loss)
    all_train_loss.append(epoch_train_loss)

    print()

    learner.eval()
    running_val_loss = 0
    count = 0 
    with torch.no_grad():
        for i,images in tqdm.tqdm(enumerate(val_data_loader)):
            images = images[0]
            count += images.shape[0]

            val_loss = learner(images)
            running_val_loss += val_loss.item() * images.shape[0]

            if i==max_batches:
                break

    epoch_val_loss = running_val_loss/(count)
    print('epoch_val_loss',epoch_val_loss)
    all_val_loss.append(epoch_val_loss)
# save your improved network

log['train_loss'] = all_train_loss
log['val_loss'] = all_val_loss

if initial_weights == 'default':    
    with open(os.path.join(log['model_path'],log['name'],'training_log.json'), 'w') as f:
        record = {}
        record[0] = log
        json.dump(record, f)
else:
    with open(os.path.join(log['initial_weights'],'training_log.json'), 'r') as f:
        record = json.load(f)
        record[len(record)] = log
    with open(os.path.join(log['model_path'],log['name'],'training_log.json'), 'w') as f:
        json.dump(record, f)

plt.plot(all_train_loss)
plt.plot(all_val_loss)
plt.legend(['training loss','validation loss'])
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy')
plt.grid()
        
plt.savefig(os.path.join(log['model_path'],log['name'],'loss.png'))
plt.clf()

torch.save(model.state_dict(), os.path.join(log['model_path'],log['name'],'ViT-Params.pt'))
torch.save(learner.state_dict(), os.path.join(log['model_path'],log['name'],'DINO-Params.pt'))

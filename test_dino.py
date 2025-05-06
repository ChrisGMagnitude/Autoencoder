import torch
from vit_pytorch import ViT,Dino
from dataLoader import MagClassDataset
import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json

model_path = r'/root/field_data/test/ml/cg/DINO Models'
train_path = r'/root/docker_data/Autoencoder/hdf5/train.hdf5'
valid_path = r'/root/docker_data/Autoencoder/hdf5/valid.hdf5'
initial_weights = r'/root/field_data/test/ml/cg/DINO Models/Run 2 DINOViT Autoencoding - high 5e-3 lr - 2025-05-02 154712'#'default'
current_time = datetime.now()
architecture = 'DINO ViT'
description = 'Run 3 DINOViT - mid 5e-4 lr - full epoch'
image_size = 416
batch_size = 128
learning_rate = 5e-4
num_epochs = 85
max_batches_train = -1
max_batches_val = -1
early_stopping_epochs = 10

train_dataset = MagClassDataset(train_path)
val_dataset = MagClassDataset(valid_path,augment=False)

print('os.cpu_count()',os.cpu_count())
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True,pin_memory=True,drop_last=True)  
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=True,pin_memory=True,drop_last=True)  

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
log['lr'] = learning_rate
log['num_epochs'] = num_epochs
log['max_batches_train'] = max_batches_train
log['max_batches_val'] = max_batches_val
log['early_stopping_epochs'] = early_stopping_epochs

os.mkdir(os.path.join(log['model_path'],log['name']))



model = ViT(
    image_size = image_size,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

if initial_weights!='default':
    model.load_state_dict(torch.load(os.path.join(initial_weights,'ViT-Params.pt'), weights_only=True))

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
model.to(device)

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

if initial_weights!='default':
    learner.load_state_dict(torch.load(os.path.join(initial_weights,'DINO-Params.pt'), weights_only=True))

learner.to(device)
    
opt = torch.optim.Adam(learner.parameters(), lr = learning_rate)

all_train_loss = []
all_val_loss = []

for epoch in range(num_epochs):
    learner.train()
    print('epoch =',epoch)
    count = 0
    running_train_loss = 0
    for i,images in enumerate(tqdm.tqdm(train_data_loader)):
        images = images[0]
        count += images.shape[0]
        images = images.to(device)

        loss = learner(images)
        running_train_loss += loss.item() * images.shape[0]
    
        opt.zero_grad()
        loss.backward()
        opt.step()
        #print('train loss',loss.item())
        if i==max_batches_train:
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
        for i,images in enumerate(tqdm.tqdm(val_data_loader)):
            images = images[0]
            images = images.to(device)
            count += images.shape[0]

            val_loss = learner(images)
            running_val_loss += val_loss.item() * images.shape[0]

            if i==max_batches_val:
                break

    epoch_val_loss = running_val_loss/(count)
    print('epoch_val_loss',epoch_val_loss)
    all_val_loss.append(epoch_val_loss)

    plt.plot(all_train_loss)
    plt.plot(all_val_loss)
    plt.legend(['training loss','validation loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Binary Cross Entropy')
    plt.grid()
        
    plt.savefig(os.path.join(log['model_path'],log['name'],str(epoch)+'_loss.png'))
    plt.clf()

    torch.save(model.state_dict(), os.path.join(log['model_path'],log['name'],'ViT-Params.pt'))
    torch.save(learner.state_dict(), os.path.join(log['model_path'],log['name'],'DINO-Params.pt'))
    # save your improved network

    if epoch_val_loss < min(all_val_loss):
        torch.save(model.state_dict(), os.path.join(log['model_path'],log['name'],'ViT-Params-minloss.pt'))
        torch.save(learner.state_dict(), os.path.join(log['model_path'],log['name'],'DINO-Params-minloss.pt'))


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

    

    if len(all_val_loss) - np.argmin(all_val_loss) > early_stopping_epochs:
        break



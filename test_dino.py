import torch
from vit_pytorch import ViT,Dino
from dataLoader import MagClassDataset
import tqdm

batch_size=256
train_dataset = MagClassDataset(r'/root/docker_data/Autoencoder/hdf5/train.hdf5')
val_dataset = MagClassDataset(r'/root/docker_data/Autoencoder/hdf5/valid.hdf5',augment=False)

train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=os.cpu_count())  
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,shuffle=True,num_workers=os.cpu_count())  

model = ViT(
    image_size = 416,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    #dim_head = 64,#169
    depth = 6,
    heads = 8,
    mlp_dim = 2048
)

learner = Dino(
    model,
    image_size = 416,
    hidden_layer = 'to_latent',        # hidden layer name or index, from which to extract the embedding
    projection_hidden_size = 256,#416,      # projector network hidden dimension
    projection_layers = 4,             # number of layers in projection network
    num_classes_K = 65336,             # output logits dimensions (referenced as K in paper)
    student_temp = 0.9,                # student temperature
    teacher_temp = 0.04,               # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
    local_upper_crop_scale = 0.4,      # upper bound for local crop - 0.4 was recommended in the paper 
    global_lower_crop_scale = 0.5,     # lower bound for global crop - 0.5 was recommended in the paper
    moving_average_decay = 0.9,        # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
    center_moving_average_decay = 0.9, # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
)

opt = torch.optim.Adam(learner.parameters(), lr = 3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 600, 600)

for epoch in range(100):
    learner.train()
    print('epoch =',epoch)
    count = 0
    running_train_loss = 0
    print('Loading Images')
    for images in tqdm.tqdm(train_data_loader):
        images = images[0]
        count += images.shape[0]
        print(count,len(train_dataset))

        print('Training model')
        loss = learner(images)
        running_train_loss += loss.item() * images.shape[0]
    
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('train loss',loss.item())
        print('Loading Images')
    learner.update_moving_average() # update moving average of teacher encoder and teacher centers
    epoch_train_loss = running_train_loss/(count*batch_size)
    print('epoch_val_loss',epoch_train_loss.item())

    print()

    learner.eval()
    running_val_loss = 0
    with torch.no_grad():
        for images in tqdm.tqdm(train_data_loader):
            val_loss = learner(sample_unlabelled_images())
            running_val_loss += val_loss.item() * images.shape[0]
    epoch_val_loss = running_val_loss/(count*batch_size)
    print('epoch_val_loss',epoch_val_loss.item())

# save your improved network
torch.save(model.state_dict(), './pretrained-net.pt')

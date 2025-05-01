import torch
from vit_pytorch import ViT,Dino

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
    images = sample_unlabelled_images()
    loss = learner(images)
    print('train loss',loss.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of teacher encoder and teacher centers

    learner.eval()
    with torch.no_grad():
        val_loss = learner(sample_unlabelled_images())
    print('val loss',val_loss.item())

# save your improved network
torch.save(model.state_dict(), './pretrained-net.pt')

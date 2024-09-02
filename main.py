from classifier_free_diffion import UNet,DDPM
from torch.utils.data import Dataset,DataLoader
import torch
import tqdm

#CUDA_VISIBLE_DEVICES=3 python main.py

class CustomDataset(Dataset):
    def __init__(self,imgs,labels):
        self.imgs=imgs.to('cuda')
        self.labels=labels.to('cuda')
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, index):
        img=self.imgs[index]
        label=self.labels[index]
        return img,label
    
model=UNet(
    dim=32,
    num_classes=10
).to('cuda')

diffusion=DDPM(
    model,
    image_size=28,
    timesteps=1000
).to('cuda')

images=torch.load('./number_data/images.pt')
labels=torch.load('./number_data/labels.pt')

dataset=CustomDataset(images,labels)
dataloader=DataLoader(dataset,batch_size=128,shuffle=True,drop_last=True)

optimizer=torch.optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.999))

pbar=tqdm.tqdm(range(20))
for epoch in pbar:
    for sub_batch in dataloader:
        sub_imgs,sub_labels=sub_batch
        optimizer.zero_grad()
        loss=diffusion(sub_imgs,classes=sub_labels)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"Loss":loss.item()})
        record=loss.item()
torch.save(model.state_dict(),'./training_weight/0821_32_1000weight.pt')

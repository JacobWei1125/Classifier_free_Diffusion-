from classifier_free_diffion import UNet,DDPM
import torch
import matplotlib.pyplot as plt

#CUDA_VISIBLE_DEVICES=3 python img_generate.py
model=UNet(
    dim=32,
    num_classes=10
).to('cuda')

diffusion=DDPM(
    model,
    image_size=28,
    timesteps=1000
).to('cuda')
model.load_state_dict(torch.load('./traing_weight/0821_32_1000weight.pt',map_location='cuda'))

image_classes = torch.arange(0,10).cuda() 
print(image_classes)
sample=diffusion.sample(
    classes=image_classes,
    cond_scale=4
)
img_tensor = sample.to('cpu') 
img_tensor = img_tensor.squeeze(1)  

for i in range(img_tensor.size(0)):
    img = img_tensor[i].numpy()  
    plt.imshow(img, cmap='gray')
    plt.title(f'Number image, {image_classes[i]}')
    plt.savefig(f'img/img_{i}.png')
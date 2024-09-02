import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange,repeat
from functools import partial
from tqdm.auto import tqdm


# U-net basic block
def Upsample(in_dims,out_dims):
    return nn.Sequential(
        nn.Upsample(scale_factor=2,mode='nearest'),
        nn.Conv2d(in_dims,out_dims,3,padding=1)
    )
def Downsample(in_dims,out_dims):
    return nn.Conv2d(in_dims,out_dims,4,2,1)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim)  
    def forward(self, x):
        return self.fn(self.norm(x))
    
# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if isinstance(x, int):  
            x = torch.tensor([x], dtype=torch.float32)
        elif x.dim() == 0:  
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
# classifier free guidance function
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob
    
# block modules

class Block(nn.Module):
    def __init__(self,in_dims,out_dims,groups=8):
        super().__init__()
        self.proj=nn.Conv2d(in_dims,out_dims,3,padding=1)
        self.norm = nn.GroupNorm(groups,out_dims)
        self.act=nn.SiLU()
    
    def forward(self,x,scale_shift=None):
        x=self.proj(x)
        x=self.norm(x)
        if scale_shift is not None:
            scale,shift=scale_shift
            x=x*(scale+1)+shift
        x=self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self,in_dims,out_dims,time_emb_dims=None,classes_emb_dims=None):
        super().__init__()
        if time_emb_dims or classes_emb_dims:
            self.mlp=nn.Sequential(
                nn.SiLU(),
                nn.Linear(int(time_emb_dims)+int(classes_emb_dims),out_dims*2)
            )
        else :
            self.mlp=None
        self.block1=Block(in_dims,out_dims)
        self.block2=Block(out_dims,out_dims)
        if in_dims != out_dims:    
            self.res_conv=nn.Conv2d(in_dims,out_dims,1)
        else:
            self.res_conv=nn.Identity()
    def forward(self,x,time_emb=None,classes_emb=None):
        expanded_time_emb = time_emb.expand(classes_emb.shape[0], -1)
        cond_emb=(expanded_time_emb,classes_emb)
        cond_emb=torch.cat(cond_emb,dim=-1)
        cond_emb=self.mlp(cond_emb)
        cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
        scale_shift = cond_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)        


class UNet(nn.Module):
    def __init__(self,dim,num_classes,channels=1,dim_mults=[1,2,4],cond_drop_prob = 0.5):    
        super().__init__()
        self.channels = channels
        self.cond_drop_prob=cond_drop_prob
        init_dim=dim    #dim=16
        self.init_conv = nn.Conv2d(self.channels, init_dim,7,padding=3)
        dims = [init_dim] + [m * dim for m in dim_mults]    #[16, 16, 32, 64]
        input_output_dims = list(zip(dims[:-1], dims[1:]))  #[(16, 16), (16, 32), (32, 64)]
    #time embeddings
        time_dim = 4 * dim
        sinu_pos_emb=SinusoidalPosEmb(dim)
        self.time_mlp=nn.Sequential(
            sinu_pos_emb,
            nn.Linear(dim,time_dim),
            nn.GELU(),
            nn.Linear(time_dim,time_dim)
        )
    #classes embedding
        self.classes_emb=nn.Embedding(num_classes,dim)
        self.null_classes_emb=nn.Parameter(torch.randn(dim))

        classes_dim=4 * dim
        self.classes_mlp=nn.Sequential(
            nn.Linear(dim,classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim,classes_dim)
        )

    #U-net layer
        self.downs=nn.ModuleList([])
        self.ups=nn.ModuleList([])
    #down sample
        for idx,(in_dim,out_dim) in enumerate(input_output_dims): #[(16, 16), (16, 32), (32, 64)]
            check_last= idx == (len(input_output_dims)-1)

            self.downs.append(nn.ModuleList([
                ResnetBlock(in_dim,in_dim,time_emb_dims=time_dim,classes_emb_dims=classes_dim),
                ResnetBlock(in_dim,in_dim,time_emb_dims=time_dim,classes_emb_dims=classes_dim),
                Residual(PreNorm(in_dim,nn.Identity())),
                Downsample(in_dim, out_dim) if not check_last else nn.Conv2d(in_dim, out_dim, 3, padding = 1)
            ]))
    #botten neck
        mid_dim=dims[-1]
        self.mid_block1=ResnetBlock(mid_dim,mid_dim,time_emb_dims=time_dim,classes_emb_dims=classes_dim)
        self.mid_res_norm=Residual(PreNorm(mid_dim,nn.Identity()))
        self.mid_block2=ResnetBlock(mid_dim,mid_dim,time_emb_dims=time_dim,classes_emb_dims=classes_dim)
    #up sample
        for idx,(in_dim,out_dim) in enumerate(reversed(input_output_dims)): #[(32, 64), (16, 32), (16, 16)]
            check_last=idx == (len(input_output_dims)-1)

            self.ups.append(nn.ModuleList([
                ResnetBlock(in_dim+out_dim,out_dim,time_emb_dims=time_dim,classes_emb_dims=classes_dim),
                ResnetBlock(in_dim+out_dim,out_dim,time_emb_dims=time_dim,classes_emb_dims=classes_dim),
                Residual(PreNorm(out_dim,nn.Identity())),
                Upsample(out_dim, in_dim) if not check_last else nn.Conv2d(out_dim, in_dim, 3, padding = 1)
            ]))

        self.final_resblock=ResnetBlock(init_dim*2,init_dim,time_emb_dims=time_dim,classes_emb_dims=classes_dim)
        self.final_conv=nn.Conv2d(init_dim,self.channels,1)
    #sample with condition
    def forward_with_cond_scale(self,*args,cond_scale=1.,rescaled_phi=0.,**kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        scaled_logits = null_logits + (logits - null_logits) * cond_scale

        if rescaled_phi == 0.:
            return scaled_logits, null_logits

        std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))
        interpolated_rescaled_logits = rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)

        return interpolated_rescaled_logits, null_logits

    def forward(self,x,time,classes,cond_drop_prob=None):
        
        batch,device=x.shape[0],x.device
        cond_drop_prob=self.cond_drop_prob

        classes_emb=self.classes_emb(classes)
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)

            classes_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                classes_emb,
                null_classes_emb
            )
        c=self.classes_mlp(classes_emb)

    #U-net      
        x=self.init_conv(x)
        r=x.clone()
        if isinstance(time, int):
            time = torch.tensor([time], dtype=torch.float32).to(device)  
        else:
            time = time.to(device)
        t=self.time_mlp(time)
        h=[]

        for block1, block2, res_norm, downsample in self.downs:
            x = block1(x, t, c)
            h.append(x)

            x = block2(x, t, c)
            x = res_norm(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, c)
        x = self.mid_res_norm(x)
        x = self.mid_block2(x, t, c)

        for block1, block2, res_norm, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, c)
            x = res_norm(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_resblock(x, t, c)
        return self.final_conv(x)

#diffusion process
def linear_schedule_beta_t(number_timestep): #alpha_t=1-bata_t
    beta_start=0.0001
    beta_end=0.02
    beta_t = torch.linspace(beta_start, beta_end, number_timestep)
    return beta_t

class DDPM(nn.Module):
    def __init__(self,model,image_size,timesteps,device='cuda'):
        super().__init__()
        self.model=model.to(device)
        self.device=device
        self.channels=self.model.channels
        self.image_size=image_size
        self.num_timesteps=timesteps
        self.beta_t=linear_schedule_beta_t(self.num_timesteps).to(device)
        self.alpha_t= 1.0-self.beta_t
        self.alphas_t_bar = torch.cumprod(self.alpha_t, dim=0).to(device)
        self.alphas_t_minus_1_bar = torch.cat((torch.tensor([0.0], device=self.device),self.alphas_t_bar[:-1])).to(device)
        self.sampling_timesteps=timesteps
    #training

    def sample_q(self, image, timestep, noise=None):
        device = self.beta_t.device
        if noise is None:
            noise = torch.randn_like(image, device=device)
        alpha_t_bar = self.alphas_t_bar[timestep].to(device).view(-1,1,1,1)
        img = torch.sqrt(alpha_t_bar) * image + torch.sqrt(1 - alpha_t_bar) * noise
        return img

    def p_losses(self,x_start,time,classes,noise=None):
        noise=torch.randn_like(x_start)
        x=self.sample_q(x_start,time,noise=noise)
        model_output=self.model(x,time,classes)
        loss=F.mse_loss(model_output,noise)
        return loss
    #sample
    def model_predictions(self,x,t,classes,cond_scale=6.,rescaled_phi=0.7):
        pred_noise,_=self.model.forward_with_cond_scale(x,t,classes,cond_scale=cond_scale,rescaled_phi=rescaled_phi)
        x_start=self.sample_q(x,t,pred_noise)

        return pred_noise,x_start
    
    @torch.no_grad()
    def sample_p(self,x_t,t,classes,cond_scale=6.,rescaled_phi=0.7):
        pred_noise,_=self.model_predictions(x_t,t,classes,cond_scale=cond_scale,rescaled_phi=rescaled_phi)
        predicted_mean=1/torch.sqrt(self.alpha_t[t])*(x_t-self.beta_t[t]/torch.sqrt(1-self.alphas_t_bar[t]) * pred_noise)
        if t==0:
            return predicted_mean
        else:
            posterior_variance=self.beta_t[t]*(1-self.alphas_t_minus_1_bar[t])/(1-self.alphas_t_bar[t])
            posterior_variance = posterior_variance.clamp(min=1e-10)
            noise=torch.randn_like(x_t)
            return predicted_mean+torch.sqrt(posterior_variance)*noise
    @torch.no_grad()
    def sample(self,classes,cond_scale=6.,rescaled_phi=0.7):
        batch,device=classes.shape[0],self.beta_t.device
        img=torch.randn(batch,1,28,28,device=device)
        for t in tqdm(reversed(range(0,self.num_timesteps)),desc = 'sampling loop time step', total = self.num_timesteps):
            img=self.sample_p(img,t,classes,cond_scale,rescaled_phi)
        img = img*2-1
        return img

    def forward(self,img,*args,**kwargs):
        
        batch_size,c,h,w,device=*img.shape,img.device
        #t=Uniform(0,num_timesteps)
        time=torch.randint(0,self.num_timesteps,(batch_size,),device=device).long()

        img = img*2-1
        return self.p_losses(img,time,*args,**kwargs)


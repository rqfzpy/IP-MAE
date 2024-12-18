import torch
import timm
import numpy as np
from torch import nn, einsum
from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import PatchEmbed,Block

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor, aspatches_shift : torch.Tensor, aspatches_mean : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))
        pam_dist = torch.abs(aspatches_shift-aspatches_mean).mean(-1)

        a = torch.topk(input =pam_dist,k=remain_T ,dim=-1,largest=False)
        b = torch.topk(input =pam_dist,k=T ,dim=-1,largest=False)

        asforward_indexes = torch.stack([i for i in a[1]], axis=-1)
        asbackward_indexes = torch.stack([i for i in b[1]], axis=-1)   
        useful_patches = patches.gather(dim=0, index=asforward_indexes.unsqueeze(-1).repeat(1,1,patches.shape[-1]))
        mask_patches = patches.gather(dim=0, index=asbackward_indexes.unsqueeze(-1).repeat(1,1,patches.shape[-1]))[remain_T:]

        return useful_patches,mask_patches, asforward_indexes, asbackward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ms =1,
                 adaptive=True
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.patch_size = patch_size
        self.adaptive = adaptive
        self.patchify= nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
                nn.Linear(patch_size ** 2*3, emb_dim),
                nn.Dropout(0.05))
        self.Rearrange= Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size)
        self.recovered_patches = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size, p2 = patch_size, h = 8, w = 8)
        if self.adaptive:
            self.conv_gamma = nn.Sequential(nn.Linear(patch_size ** 2*3, patch_size ** 2*3)
            ,nn.Sigmoid(),nn.Linear(patch_size ** 2*3, 1))
            self.conv_beta= nn.Sequential(nn.Linear(patch_size ** 2*3, patch_size ** 2*3)
            ,nn.Sigmoid(),nn.Linear(patch_size ** 2*3, 1))
        else:
            self.conv_gamma = nn.Sequential(nn.Linear(patch_size ** 2*3, patch_size ** 2*3)
            ,nn.Sigmoid())

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)
        self.ms = ms

        if ms ==1:
            self.Rearrange_2= Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size*2, p2 = patch_size*2)
            self.recovered_patches_2 = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size*2, p2 = patch_size*2, h = 4, w = 4)

        if ms ==2:
            self.Rearrange_2= Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size*2, p2 = patch_size*2)
            self.recovered_patches_2 = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size*2, p2 = patch_size*2, h = 4, w = 4)
            self.Rearrange_3= Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size*4, p2 = patch_size*4)
            self.recovered_patches_3 = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size*4, p2 = patch_size*4, h = 2, w = 2)
        if ms ==3:
            self.Rearrange_2= Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size*2, p2 = patch_size*2)
            self.recovered_patches_2 = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size*2, p2 = patch_size*2, h = 4, w = 4)
            self.Rearrange_3= Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size*4, p2 = patch_size*4)
            self.recovered_patches_3 = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1 = patch_size*4, p2 = patch_size*4, h = 2, w = 2)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img,label,num_class):
        
        eps = 1e-6
        centroids = torch.zeros((num_class, img.shape[1], img.shape[2], img.shape[3])).to('cuda')
        aspatches = self.Rearrange(img)
        pixel_var = torch.zeros((num_class, aspatches.shape[1],aspatches.shape[2])).to('cuda')
        vars = torch.zeros((num_class, aspatches.shape[1])).to('cuda')

        for i in range(num_class):
            class_data = img[label == i]
            centroids[i] = torch.mean(class_data, dim=0)
            var_data = aspatches[label == i]
            pixel_var[i] = var_data.var(dim=0,unbiased=True)
            vars[i] = var_data.var(dim=0,unbiased=True).mean(dim=-1)
        pixel_var[torch.isnan(pixel_var)] = eps
        vars[torch.isnan(vars)] = eps

        new_img = torch.zeros_like(img).to('cuda')
        pixel_var_patch = torch.zeros(aspatches.shape[0],pixel_var.shape[1],pixel_var.shape[2]).to('cuda')
        var_patch = torch.zeros(aspatches.shape[0],vars.shape[1]).to('cuda')

        for i in range(img.shape[0]):
            class_id = label[i]
            new_img[i] = centroids[class_id]
            var_patch[i] = vars[class_id]
            pixel_var_patch[i] = pixel_var[class_id]

        patches = self.patchify(img)
    
        aspatches_mean = self.Rearrange(new_img)
        patches = rearrange(patches, 'b t c -> t b c')
        patches = patches + self.pos_embedding
        if self.adaptive:
            gamma_shift = self.conv_gamma(aspatches)
            beta_shift = self.conv_beta(aspatches)
            aspatches_shift = aspatches*gamma_shift+beta_shift
        else:
            aspatches_shift = self.conv_gamma(aspatches)

        if self.ms ==0:
            var0 = pixel_var_patch
            total_var = [var0]
        if self.ms ==1:
            var0 = pixel_var_patch
            var1 = var_patch.unsqueeze(-1).expand(-1, -1, aspatches.shape[-1])
            total_var = [var0, var1]
        if self.ms ==2:
            var0 = pixel_var_patch
            var1 = var_patch.unsqueeze(-1).expand(-1, -1, aspatches.shape[-1])
            aspatches_shift = self.recovered_patches(aspatches_shift)
            patches_2 = self.Rearrange_2(img)
            vars_2 = torch.zeros((num_class, patches_2.shape[1])).to('cuda')
            for i in range(num_class):
                var2_data = patches_2[label == i]
                vars_2[i] = var2_data.var(dim=0,unbiased=True).mean(dim=-1)
            vars_2[torch.isnan(vars_2)] = eps
            vars2_patch = torch.zeros(patches_2.shape[0],vars_2.shape[1]).to('cuda')

            for i in range(patches_2.shape[0]):
                class_id = label[i]
                vars2_patch[i] = vars_2[class_id]
            aspatches_shift_2 = self.Rearrange_2(aspatches_shift)
            var2 = self.Rearrange(self.recovered_patches_2(vars2_patch.unsqueeze(-1).expand(-1, -1, aspatches_shift_2.shape[-1])))
            aspatches_shift = self.Rearrange(aspatches_shift)
            total_var = [var0, var1, var2]
        if self.ms ==3:
            var0 = pixel_var_patch
            var1 = var_patch.unsqueeze(-1).expand(-1, -1, aspatches.shape[-1])
            aspatches_shift = self.recovered_patches(aspatches_shift)
            patches_2 = self.Rearrange_2(img)
            vars_2 = torch.zeros((num_class, patches_2.shape[1])).to('cuda')
            for i in range(num_class):
                var2_data = patches_2[label == i]
                vars_2[i] = var2_data.var(dim=0,unbiased=True).mean(dim=-1)
            vars_2[torch.isnan(vars_2)] = eps
            vars2_patch = torch.zeros(patches_2.shape[0],vars_2.shape[1]).to('cuda')

            for i in range(patches_2.shape[0]):
                class_id = label[i]
                vars2_patch[i] = vars_2[class_id]
            aspatches_shift_2 = self.Rearrange_2(aspatches_shift)
            var2 = self.Rearrange(self.recovered_patches_2(vars2_patch.unsqueeze(-1).expand(-1, -1, aspatches_shift_2.shape[-1])))
            aspatches_shift = self.Rearrange(aspatches_shift)
            aspatches_shift = self.recovered_patches(aspatches_shift)
            patches_2 = self.Rearrange_2(img)
            patches_3 = self.Rearrange_3(img)
            vars_2 = torch.zeros((num_class, patches_2.shape[1])).to('cuda')
            vars_3 = torch.zeros((num_class, patches_3.shape[1])).to('cuda')
            for i in range(num_class):
                var2_data = patches_2[label == i]
                vars_2[i] = var2_data.var(dim=0,unbiased=True).mean(dim=-1)
                var3_data = patches_3[label == i]
                vars_3[i] = var3_data.var(dim=0,unbiased=True).mean(dim=-1)
            vars_2[torch.isnan(vars_2)] = eps
            vars_3[torch.isnan(vars_3)] = eps
            vars2_patch = torch.zeros(patches_2.shape[0],vars_2.shape[1]).to('cuda')
            vars3_patch = torch.zeros(patches_3.shape[0],vars_3.shape[1]).to('cuda')
            for i in range(patches_2.shape[0]):
                class_id = label[i]
                vars2_patch[i] = vars_2[class_id]
                vars3_patch[i] = vars_3[class_id]
            aspatches_shift_3 = self.Rearrange_3(aspatches_shift)
            aspatches_shift = self.Rearrange(aspatches_shift)
            var3 = self.Rearrange(self.recovered_patches_3(vars3_patch.unsqueeze(-1).expand(-1, -1, aspatches_shift_3.shape[-1])))
            total_var = [var0, var1, var2, var3]

        patches, mask_patches,forward_indexes, backward_indexes = self.shuffle(patches,aspatches_shift,aspatches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features,mask_patches, backward_indexes,aspatches_mean,aspatches_shift,total_var

class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 decoder =False,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)
        self.decoder =decoder
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features,mask_patches, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        if self.decoder:
            features = torch.cat([features, mask_patches], dim=0)
        else:
            features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=4,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 decoder =False,
                 ms = 1,
                 adaptive=True,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio,ms,adaptive)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head,decoder)

    def forward(self, img,label,num_class):
        features,mask_patches, backward_indexes,aspatches_mean,aspatches_shift,total_var = self.encoder(img,label,num_class)
        predicted_img, mask = self.decoder(features, mask_patches, backward_indexes)
        return predicted_img, mask,aspatches_mean,aspatches_shift,total_var

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder : MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b t c -> t b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits
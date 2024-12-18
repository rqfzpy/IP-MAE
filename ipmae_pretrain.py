import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm
from utils.dataloader import datainfo, dataload
import torchvision.transforms as transforms
from colorama import Fore, Style
from utils.sampler import RASampler
import torch.nn.functional as F
from thop import profile
from ipmae_model import *
from util import setup_seed
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=128)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=200)
    parser.add_argument('--warmup_epoch', type=int, default=50)
    parser.add_argument('--model_path', type=str, default='ip-mae.pt')
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--data_path', default='/mnt/data/dataset', type=str, help='dataset path')
    parser.add_argument('--dataset', default='CIFAR100', type=str, help='Image Net dataset path')
    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    parser.add_argument('--decoder', action='store_true', help='decoder')
    parser.add_argument('--ms', type=int, default=0, help='ms')
    parser.add_argument('--adaptive', action='store_false', help='adaptive')
    parser.add_argument('--re', default=0.25, type=float, help='Random Erasing probability')
    parser.add_argument('--re_sh', default=0.4, type=float, help='max erasing area')
    parser.add_argument('--re_r1', default=0.3, type=float, help='aspect of erasing area')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    if args.adaptive:
        if args.decoder:
            save_path = os.path.join(os.getcwd(), 'ipmae-save-pretrain-adaptive-decoder'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))
        else:
            save_path = os.path.join(os.getcwd(), 'ipmae-save-pretrain-adaptive'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))
        if save_path:
            os.makedirs(save_path, exist_ok=True)
    else:
        if args.decoder:
            save_path = os.path.join(os.getcwd(), 'ipmae-save-pretrain-decoder'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))
        else:
            save_path = os.path.join(os.getcwd(), 'ipmae-save-pretrain'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))
        if save_path:
            os.makedirs(save_path, exist_ok=True)

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    data_info = datainfo(args)
    '''
        Data Augmentation
    '''
    normalize = [transforms.Normalize(mean=data_info['stat'][0], std=data_info['stat'][1])]

    augmentations = []
    
    
    augmentations += [   
            transforms.Resize(data_info['img_size']),             
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(data_info['img_size'], padding=4)
            ]
    
    
    if args.aa == True:
        print(Fore.YELLOW+'*'*80)    
        
        if 'CIFAR' in args.dataset:
            print("CIFAR Policy")
            from utils.autoaug import CIFAR10Policy
            augmentations += [   
                CIFAR10Policy()
            ]
            
        elif 'SVHN' in args.dataset:
            print("SVHN Policy")    
            from utils.autoaug import SVHNPolicy
            augmentations += [
                SVHNPolicy()
            ]
                    
        else:
            from utils.autoaug import ImageNetPolicy
            augmentations += [                
                ImageNetPolicy()
            ]
            
        print('*'*80 + Style.RESET_ALL)
    
    augmentations += [                
            transforms.ToTensor(),
            *normalize]  
    if args.re > 0:
        from utils.random_erasing import RandomErasing
        print(Fore.YELLOW + '*'*80)
        print('*'*80+Style.RESET_ALL)    
        
        augmentations += [     
            RandomErasing(probability = args.re, sh = args.re_sh, r1 = args.re_r1, mean=data_info['stat'][0])
            ]

    augmentations = transforms.Compose(augmentations)

    train_dataset, val_dataset = dataload(args, augmentations, normalize, data_info)
    dataloader = torch.utils.data.DataLoader(
        train_dataset,  num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    if args.dataset == 'T-IMNET':
        model = MAE_ViT(image_size=data_info['img_size'],patch_size=8,mask_ratio=args.mask_ratio,decoder=args.decoder,ms=args.ms,adaptive=args.adaptive).to(device)
    else:
        model = MAE_ViT(mask_ratio=args.mask_ratio,decoder=args.decoder,ms=args.ms,adaptive=args.adaptive).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    step_count = 0
    optim.zero_grad()
    epoch_losses = []
    for e in range(args.total_epoch):
        model.train()
        losses = []
        ce_losses =[]
        ip_losses =[]
        for img, label in tqdm(iter(dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            predicted_img, mask,aspatches_mean,aspatches_shift,total_var = model(img,label,data_info['n_classes'])
            ce_loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
            vwa_loss = torch.abs(aspatches_shift - aspatches_mean)
            ip_loss = 0.
            for i in range(args.ms + 1): 
                softmax_weights = torch.stack([F.softmax(1 / (var+0.5), dim=0) for var in total_var])
                weighted_loss = vwa_loss * softmax_weights
                ip_loss += torch.mean(weighted_loss)
            loss = ce_loss+ip_loss
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            ce_losses.append(ce_loss.item())
            ip_losses.append(ip_loss.item())
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        avg_ce_loss = sum(ce_losses) / len(losses)
        avg_ip_loss = sum(ip_losses) / len(losses)
        epoch_losses.append({
        'epoch': e,
        'avg_loss': avg_loss,
        'avg_ce_loss': avg_ce_loss,
        'avg_ip_loss': avg_ip_loss})
        print(f'In epoch {e}, average traning loss is {avg_loss}, average ce loss is {avg_ce_loss}, average traning loss is {avg_ip_loss}.')
      

        with open('epoch_losses.json', mode='w') as file:
            json.dump(epoch_losses, file, indent=4)
        
        ''' save model '''
        torch.save(model, save_path+'/'+args.dataset+'-'+args.model_path)
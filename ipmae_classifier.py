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
import json
from ipmae_model import *
from util import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=300)
    parser.add_argument('--mask_ratio', type=float, default=0.15)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='as-vit-t-classifier-from_scratch.pt')
    parser.add_argument('--output_model_finetuning_path', type=str, default='vit-t-classifier-finetuning.pt')
    parser.add_argument('--model_path', type=str, default='ip-mae.pt')

    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--data_path', default='/mnt/data/dataset', type=str, help='dataset path')
    parser.add_argument('--dataset', default='T-IMNET', choices=['CIFAR10', 'CIFAR100', 'T-IMNET', 'IMNET', 'SVHN'], type=str, help='Image Net dataset path')
    parser.add_argument('--aa', action='store_false', help='Auto augmentation used'),
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--ra', type=int, default=3, help='repeated augmentation')
    parser.add_argument('--decoder', action='store_true', help='decoder')
    parser.add_argument('--adaptive', action='store_false', help='adaptive')
    parser.add_argument('--ms', type=int, default=4, help='ms')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

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
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(data_info['img_size'])
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

    augmentations = transforms.Compose(augmentations)

    train_dataset, val_dataset = dataload(args, augmentations, normalize, data_info)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,  num_workers=args.workers, pin_memory=True,
        batch_sampler=RASampler(len(train_dataset), args.batch_size, 1, args.ra, shuffle=True, drop_last=True))
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=args.workers)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        if args.adaptive:
            if args.decoder:
                model = torch.load(os.path.join(os.getcwd(), 'ipmae-save-pretrain-adaptive-decoder'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))+'/'+args.dataset+'-'+args.model_path, map_location='cpu')
                save_path = os.path.join(os.getcwd(), 'ipmae-save-funetuning-adaptive-decoder'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
            else:
                model = torch.load(os.path.join(os.getcwd(), 'ipmae-save-pretrain-adaptive'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))+'/'+args.dataset+'-'+args.model_path, map_location='cpu')
                save_path = os.path.join(os.getcwd(), 'ipmae-save-funetuning-adaptive'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
        else:
            if args.decoder:
                model = torch.load(os.path.join(os.getcwd(), 'ipmae-save-pretrain-decoder'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))+'/'+args.dataset+'-'+args.model_path, map_location='cpu')
                save_path = os.path.join(os.getcwd(), 'ipmae-save-funetuning-decoder'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
            else:
                model = torch.load(os.path.join(os.getcwd(), 'ipmae-save-pretrain'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))+'/'+args.dataset+'-'+args.model_path, map_location='cpu')
                save_path = os.path.join(os.getcwd(), 'ipmae-save-funetuning'+'-mr'+str(args.mask_ratio*100)+'-ms-'+str(args.ms))
                if save_path:
                    os.makedirs(save_path, exist_ok=True)
    else:
        if args.dataset == 'T-IMNET':
            model = MAE_ViT(image_size=data_info['img_size'],patch_size=8,mask_ratio=args.mask_ratio).to(device)
        else:
            model = MAE_ViT(mask_ratio=args.mask_ratio).to(device)
        save_path = os.path.join(os.getcwd(), 'ipmae-save-scratch')
        if save_path:
            os.makedirs(save_path, exist_ok=True)
    model = ViT_Classifier(model.encoder, num_classes=data_info['n_classes']).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()
    results = []
    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        for img, label in tqdm(iter(train_dataloader)):
            step_count += 1
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            loss = loss_fn(logits, label)
            acc = acc_fn(logits, label)
            loss.backward()
            if step_count % steps_per_update == 0:
                optim.step()
                optim.zero_grad()
            losses.append(loss.item())
            acces.append(acc.item())
        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            for img, label in tqdm(iter(val_dataloader)):
                img = img.to(device)
                label = label.to(device)
                logits = model(img)
                loss = loss_fn(logits, label)
                acc = acc_fn(logits, label)
                losses.append(loss.item())
                acces.append(acc.item())
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  
            results.append({
            "epoch": e,
            "average_validation_loss": avg_val_loss,
            "average_validation_accuracy": avg_val_acc
        })

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!') 
            if args.pretrained_model_path is not None:  
                torch.save(model, save_path+'/'+args.dataset+'-'+args.output_model_finetuning_path)
            else:    
                torch.save(model, save_path+'/'+args.dataset+'-'+args.output_model_path)
    output_file = "validation_results.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
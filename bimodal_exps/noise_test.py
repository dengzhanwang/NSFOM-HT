import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from models.model_clip import CLIP
from dataset import create_train_dataset, create_val_dataset, create_sampler, create_train_loader, create_val_loader
from transformers import AutoTokenizer

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

def noise_test(model, dataloader, device):
    """
    检测模型在数据集上梯度的噪声。
    
    参数:
        model: 训练好的模型。
        dataloader: 数据加载器（如 val_coco_loader 或 val_flickr_loader）。
        device: 设备（如 'cuda' 或 'cpu'）。
    
    返回:
        noise_norms: 每个 minibatch 梯度与全梯度的差异的模长列表。
    """
    model = model.to(device)
    model.train()  # 设置为训练模式以计算梯度

    # 获取模型参数
    params = [p for p in model.parameters() if p.requires_grad]

    # 计算全梯度
    print("Calculating full gradient...")
    full_gradients = [torch.zeros_like(p) for p in params]  # 初始化全梯度
    total_samples = 0

    for batch in tqdm(dataloader):
        images, texts, imd, index = batch
        images = images.to(device)
        text_input = tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)

        # 计算损失
        loss, avg_image_tau, avg_text_tau, cur_eta, grad_tau_image, grad_tau_text, b_I, b_T = model(images, text_input,imd,index,epoch=0,max_epoch=0,iter=0)
        loss.backward()

        # 累加梯度
        for i, p in enumerate(params):
            if p.grad is not None:
                full_gradients[i] += p.grad * len(images)  # 乘以 batch size
                p.grad.zero_()  # 清空梯度

        total_samples += len(images)

    # 计算平均梯度
    for i in range(len(full_gradients)):
        full_gradients[i] /= total_samples

    # 计算 minibatch 梯度与全梯度的差异
    print("Calculating minibatch gradient noise...")
    noise_norms = []

    for batch in tqdm(dataloader):
        images, texts, _, _ = batch
        images = images.to(device)
        text_input = tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)

        # 计算 minibatch 梯度
        loss, avg_image_tau, avg_text_tau, cur_eta, grad_tau_image, grad_tau_text, b_I, b_T = model(images, text_input,imd,index,epoch=0,max_epoch=0,iter=0)
        loss.backward()

        # 计算梯度差异
        noise = []
        for i, p in enumerate(params):
            if p.grad is not None:
                noise.append(p.grad - full_gradients[i])  # 计算差异
                p.grad.zero_()  # 清空梯度

        # 计算差异的模长
        noise_norm = torch.norm(torch.cat([n.view(-1) for n in noise])).item()
        noise_norms.append(noise_norm)

    return noise_norms

import torch
from tqdm import tqdm

def gradient_noise(model1, model2, dataloader, device):
    """
    计算两个不同模型在同一个 batch 上的梯度噪声。
    
    参数:
        model1: 第一个模型。
        model2: 第二个模型。
        dataloader: 数据加载器（如 val_coco_loader 或 val_flickr_loader）。
        device: 设备（如 'cuda' 或 'cpu'）。
    
    返回:
        noise_norms: 每个 minibatch 两个模型梯度的差异的模长列表。
    """
    model1 = model1.to(device)
    model2 = model2.to(device)
    model1.train()  # 设置为训练模式以计算梯度
    model2.train()  # 设置为训练模式以计算梯度

    # 获取模型参数
    params1 = [p for p in model1.parameters() if p.requires_grad]
    params2 = [p for p in model2.parameters() if p.requires_grad]

    # 计算 minibatch 梯度噪声
    print("Calculating gradient noise...")
    noise_norms = []

    for batch in tqdm(dataloader):
        images, texts, imd, index = batch
        images = images.to(device)
        text_input = tokenizer(texts, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)

        # 计算 model1 的梯度
        loss1, _, _, _, _, _, _, _ = model1(images, text_input, imd, index, epoch=0, max_epoch=0, iter=0)
        loss1.backward()
        gradients1 = [p.grad.clone() for p in params1 if p.grad is not None]
        for p in params1:
            if p.grad is not None:
                p.grad.zero_()  # 清空梯度

        # 计算 model2 的梯度
        loss2, _, _, _, _, _, _, _ = model2(images, text_input, imd, index, epoch=0, max_epoch=0, iter=0)
        loss2.backward()
        gradients2 = [p.grad.clone() for p in params2 if p.grad is not None]
        for p in params2:
            if p.grad is not None:
                p.grad.zero_()  # 清空梯度

        # 计算梯度差异
        noise = []
        for g1, g2 in zip(gradients1, gradients2):
            noise.append(g1 - g2)  # 计算差异

        # 计算差异的模长
        noise_norm = torch.norm(torch.cat([n.view(-1) for n in noise])).item()
        noise_norms.append(noise_norm)

    return noise_norms

def plot_noise_distributions(noise_norms, fig_dir):
    """
    绘制梯度噪声范数的直方图和QQ图。

    参数:
    - noise_norms: List[float]，梯度噪声范数的列表。
    - fig_dir: str，保存图表的目录路径。
    - n_layer: int，图层编号（用于文件名）。
    - N: int，样本数量（用于文件名）。
    """
    # 设置绘图风格
    # plt.style.use('seaborn-darkgrid')
    
    # 创建主图：直方图
    fig, axs = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    
    # 直方图
    ax = axs[0]
    bins = 100
    ax.hist(noise_norms, bins=bins, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title('Gradient Noise Norms Histogram', fontsize=16)
    ax.set_xlabel('Gradient Error', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # QQ图
    ax = axs[1]
    osm, osr = stats.probplot(noise_norms, dist="norm", plot=ax)
    ax.set_title('Gradient Noise Norms QQ Plot', fontsize=16)
    ax.set_xlabel('Theoretical Quantiles', fontsize=14)
    ax.set_ylabel('Sample Quantiles', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 美化图形
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    # 保存图形
    filename = f'{fig_dir}/gradient_noise_histogram_qq.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图表已保存至 {filename}")
    
    # 保存图形
    filename = f'{fig_dir}/gradient_noise_histogram_qq.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"图表已保存至 {filename}")

           
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # data path
    parser.add_argument('--data_path', default='./dataset/')
    parser.add_argument('--train_file', default='cc3m/cc3m_train/combined2.json')
    parser.add_argument('--train_image_root', default='cc3m/cc3m_train/train')
    # parser.add_argument('--train_file', default='flickr30k/flickr30k_train.json')
    # parser.add_argument('--train_image_root', default='flickr30k/flickr30k-images')
    # parser.add_argument('--train_file', default='coco/coco_train_new.json')
    # parser.add_argument('--train_image_root', default='coco/')
    # model config
    parser.add_argument('--bert_config', default='configs/config_bert.json')
    parser.add_argument('--image_encoder', default='resnet50')
    parser.add_argument('--text_encoder', default='distilbert-base-uncased')
    parser.add_argument('--image_res', default=256)
    parser.add_argument('--vision_width', default=768)
    parser.add_argument('--embed_dim', default=256)

    # optimizer and schedular
    parser.add_argument('--opt', default='adamW')
    parser.add_argument('--sched', default='cosine')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--min_lr', default=1e-5, type=float)
    parser.add_argument('--warmup', default=True, type=bool)
    parser.add_argument('--warmup_lr', default=1e-5)
    parser.add_argument('--weight_decay', default=0.02)
    parser.add_argument('--decay_rate', default=1)
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--warmup_epochs', default=20)
    parser.add_argument('--cooldown_epochs', default=0)

    # training & test settings
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--init_model', action='store_true')
    parser.add_argument('--batch_size_train', default=256)
    parser.add_argument('--batch_size_test', default=256)
    parser.add_argument('--k_test', default=256)
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)

    # output path
    parser.add_argument('--output_dir', default='./output/clip_test')  

    # loss config
    parser.add_argument('--ita_type', default='clip', choices=['clip','rankclip' ,'cyclip', 'vicreg', 'sogclr', 'sogclr_dro'])
    parser.add_argument('--vicreg_sim_coeff', default=25.0, type=float)
    parser.add_argument('--vicreg_std_coeff', default=25.0, type=float)
    parser.add_argument('--sogclr_gamma', default=0.8, type=float)
    parser.add_argument('--rho_init', default=8.0, type=float)
    parser.add_argument('--eta_init', default=0.001, type=float)
    parser.add_argument('--tau_init', default=0.01, type=float)
    parser.add_argument('--eta_sched', choices=['const', 'cosine', 'exp'])
    parser.add_argument('--eta_exp_gamma', default=0.8, type=float)
    parser.add_argument('--beta_u', default=0.9, type=float)
    parser.add_argument('--temp', default=0.01, type=float)
    parser.add_argument('--learnable_temp', action='store_true')
    parser.add_argument('--personalized_tau', action='store_true')
    parser.add_argument('--enable_surrogate', action='store_true')

    # zero-shot transfer
    parser.add_argument('--zs_dataset', default= 'cifar10', choices=['cifar10', 'cifar100', 'imagenet'])
    parser.add_argument('--zs_datafolder', default='./datasets', type=str)

    args = parser.parse_args()

    args.train_file = os.path.join(args.data_path, args.train_file)
    args.train_image_root = os.path.join(args.data_path, args.train_image_root)

    args.val_coco_file = os.path.join(args.data_path, 'clip_train/coco_val_new.json')
    args.test_coco_file = os.path.join(args.data_path, 'clip_train/coco_test_new.json')
    args.coco_image_root = os.path.join(args.data_path, 'coco')
    args.val_flickr_file = os.path.join(args.data_path, 'clip_train/flickr30k_val.json')
    args.test_flickr_file = os.path.join(args.data_path, 'clip_train/flickr30k_test.json')
    args.flickr_image_root = os.path.join(args.data_path, 'flickr30k/flickr30k-images')


    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    json.dump(args.__dict__, open(os.path.join(args.output_dir, 'args.json'), 'w'), indent=2) 
    tokenizer = AutoTokenizer.from_pretrained('./models/bert', local_files_only=True)

    # 加载模型
    
    model = CLIP(
        image_encoder=args.image_encoder,
        text_encoder='./models/bert',
        embed_dim=args.embed_dim,
        init_model=args.init_model,
        bsz=args.batch_size_train * args.world_size,
        world_size=args.world_size,
        ita_type=args.ita_type,
        sogclr_gamma=args.sogclr_gamma,
        rho_init=args.rho_init,
        tau_init=args.tau_init,
        eta_init=args.eta_init,
        eta_sched=args.eta_sched,
        eta_exp_gamma=args.eta_exp_gamma,
        beta_u=args.beta_u,
        temp=args.temp,
        learnable_temp=args.learnable_temp,
        vicreg_sim_coeff=args.vicreg_sim_coeff,
        vicreg_std_coeff=args.vicreg_std_coeff,
        personalized_tau=args.personalized_tau,
        enable_surrogate=args.enable_surrogate,
        len_train = 10
    )

    model2 = CLIP(
        image_encoder=args.image_encoder,
        text_encoder='./models/bert',
        embed_dim=args.embed_dim,
        init_model=args.init_model,
        bsz=args.batch_size_train * args.world_size,
        world_size=args.world_size,
        ita_type=args.ita_type,
        sogclr_gamma=args.sogclr_gamma,
        rho_init=args.rho_init,
        tau_init=args.tau_init,
        eta_init=args.eta_init,
        eta_sched=args.eta_sched,
        eta_exp_gamma=args.eta_exp_gamma,
        beta_u=args.beta_u,
        temp=args.temp,
        learnable_temp=args.learnable_temp,
        vicreg_sim_coeff=args.vicreg_sim_coeff,
        vicreg_std_coeff=args.vicreg_std_coeff,
        personalized_tau=args.personalized_tau,
        enable_surrogate=args.enable_surrogate,
        len_train = 10
    ) 
    checkpoint_path = './output/clip_test/checkpoint_epoch_1.pth'  # 替换为你的检查点路径
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()  # 设置为评估模式

    checkpoint_path = './output/clip_test/checkpoint_epoch_2.pth'  # 替换为你的检查点路径
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model2.load_state_dict(checkpoint["model"])
    model2.eval()  # 设置为评估模式
    samplers = [None, None, None]
    train_dataset = create_train_dataset('re', args)
    # val_flickr_dataset, test_flickr_dataset = create_train_dataset('re', args, args.val_flickr_file, args.test_flickr_file, args.flickr_image_root)
    # 加载数据集
    train_loader = create_train_loader(train_dataset, samplers[0], args.batch_size_train, 8, None)
    # val_flickr_loader, test_flickr_loader = create_train_loader(
    #     [val_flickr_dataset, test_flickr_dataset], samplers[1:], 
    #     [args.batch_size_test] * 2, [8] * 2, [None] * 2
    # )

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 调用 noise_test 函数
    # noise_norms = noise_test(model, train_loader, device)
    noise_norms = gradient_noise(model, model2, train_loader, device)
    np.save('gradient_noise_norms.cc3m2', noise_norms)

    plot_noise_distributions(noise_norms, fig_dir=args.output_dir)
    # 打印结果
    print("Gradient noise norms:", noise_norms)
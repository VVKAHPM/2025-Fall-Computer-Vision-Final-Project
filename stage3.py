import random
import numpy as np
import torch
import cv2
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet50, ResNet50_Weights
import os
from tqdm import tqdm
from explainers import CAM, GradCAM
from PIL import Image
import argparse
from utils import save_visual_result, plot_deletion_metric, plot_attack_comparison, plot_attack_curves

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
weights = ResNet50_Weights.DEFAULT
categories = weights.meta["categories"]

target_photos = None

def get_args():
    parser = argparse.ArgumentParser(description="Stage 3: Grad-CAM guided PGD")

    parser.add_argument('--path', type=str, help='image path')
    parser.add_argument('--target', type=int, default=42, help="target index")
    return parser.parse_args()

def texture_hijacking_attack(model, img, target, mask, eps=12/255, alpha=None, iters=50, orig_label=0, method="cosine", explainer=None):
    # assume image is original file
    new_img = img.clone().to(device)
    # noise = torch.empty_like(img).uniform_(-eps, eps).to(device)
    # new_img += noise
    new_img = torch.clamp(new_img, 0, 1)


    mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(device)

    if alpha == None:
        if method == "cosine":
            alpha = eps / 4
        else:
            alpha = eps / iters
    
    target_tensor = torch.tensor([target]).to(device)
    history = {'orig_logit': [], 'target_logit': [], 'orig_prob': [], 'target_prob': []}

    for i in range(iters):
        if method == "cosine":
            curr_alpha = 0.1 / 255 + 0.5 * (alpha - 0.1 / 255) * (1 + np.cos(np.pi * i / iters))
        else:
            curr_alpha = alpha
        new_img.requires_grad = True
        input_tensor = (new_img - mean) / std
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        history['orig_logit'].append(outputs[0, orig_label].item())
        history['target_logit'].append(outputs[0, target].item())
        history['orig_prob'].append(probs[0, orig_label].item())
        history['target_prob'].append(probs[0, target].item())
        if probs[0, target] > 0.7:
            break
        loss = F.cross_entropy(outputs, target_tensor)
        grad = torch.autograd.grad(loss, new_img)[0]

        with torch.no_grad():
            new_img = new_img - curr_alpha * grad.sign() * mask.detach()
            delta = torch.clamp(new_img - img, -eps, eps)
            new_img = torch.clamp(img + delta, 0, 1).detach()

    

    return new_img, history

def get_prototype(target, folder_path='./imagenet-sample-images'):
    cnt = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.JPEG'):
            if cnt == target:
                print(filename)
                img_path = os.path.join(folder_path, filename)
                img = Image.open(img_path).convert('RGB')
                return img, target 
            cnt += 1
            
    print("Failed to find image")
    return None, None

def get_mask(heatmap, method, ratio=0.1):
    heatmap_tensor = torch.from_numpy(heatmap)
    heatmap_tensor = heatmap_tensor.unsqueeze(0).unsqueeze(0)
    heatmap_tensor = F.interpolate(heatmap_tensor, size=(224, 224), mode='bicubic', align_corners=False)
    heatmap_tensor = ((heatmap_tensor - heatmap_tensor.min()) / (heatmap_tensor.max() - heatmap_tensor.min() + 1e-7)).clamp(0, 1)
    h, w = heatmap_tensor.shape[-2:]
    total_pixels = h * w
    num = int(total_pixels * ratio)
    flat_heatmap = heatmap_tensor.view(-1)
    if method == "quantile":
        _, indices = torch.topk(flat_heatmap, k=num, largest=True, sorted=False)
        mask_flat = torch.zeros(total_pixels)
        mask_flat[indices] = 1.0
        mask = mask_flat.view(h, w)
    if method == "quantileinverse":
        _, indices = torch.topk(flat_heatmap, k=num, largest=False, sorted=False)
        mask_flat = torch.zeros(total_pixels)
        mask_flat[indices] = 1.0
        mask = mask_flat.view(h, w)
    if method == "heatmap":
        mask = heatmap_tensor
    if method == "PGD":
        mask = torch.ones_like(heatmap_tensor)
    if method == "heatmap2":
        mask = torch.pow(heatmap_tensor, 2)
    if method == "inverse":
        mask = 1 - heatmap_tensor
    if method == "inverse2":
        mask = torch.pow(1 - heatmap_tensor, 2)
    if method == "random": 
        mask_flat = torch.zeros(total_pixels)
        indices = torch.linspace(0, total_pixels - 1, steps=num).long()
        mask_flat[indices] = 1.0
        mask = mask_flat.view(heatmap_tensor.shape)
    return mask

def attack(model, categories, raw_img, target, explainer, proto_heatmap, flag=False):
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    if not flag:
        img = preprocess(raw_img).clone()
    else:
        img = raw_img.clone().squeeze(0)
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    normalize_img = normalize(img).unsqueeze(0).to(device)
    out_orig = model(normalize_img)
    pred_orig_idx = out_orig.argmax(1).item()
    heatmap = explainer.generate(normalize_img, pred_orig_idx)
    save_visual_result(normalize_img, heatmap, categories[pred_orig_idx], "dog", 0, T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ))
    
    
    img = img.unsqueeze(0).to(device)
    # mask1 = get_mask(heatmap, method="heatmap2").to(device)
    # new_img1, history1 = texture_hijacking_attack(model, img, target, mask1, orig_label=pred_orig_idx)
    # mask2 = get_mask(heatmap, method="heatmap").to(device)
    # new_img2, history2 = texture_hijacking_attack(model, img, target, mask2, orig_label=pred_orig_idx)
    mask3 = get_mask(heatmap, method="PGD").to(device)
    new_img3, history3 = texture_hijacking_attack(model, img, target, mask3, orig_label=pred_orig_idx)
    # mask4 = get_mask(heatmap, method="inverse").to(device)
    # new_img4, history4 = texture_hijacking_attack(model, img, target, mask4, orig_label=pred_orig_idx)
    # mask5 = get_mask(heatmap, method="inverse2").to(device)
    # new_img5, history5 = texture_hijacking_attack(model, img, target, mask5, orig_label=pred_orig_idx)
    mask6 = get_mask(heatmap, method="quantile").to(device)
    new_img6, history6 = texture_hijacking_attack(model, img, target, mask6, orig_label=pred_orig_idx)
    mask7 = get_mask(heatmap, method="quantileinverse").to(device)
    new_img7, history7 = texture_hijacking_attack(model, img, target, mask7, orig_label=pred_orig_idx)
    mask8 = get_mask(heatmap, method="random").to(device)
    new_img8, history8 = texture_hijacking_attack(model, img, target, mask8, orig_label=pred_orig_idx)
    
    mask9 = get_mask(proto_heatmap, method="quantile", ratio=0.2).to(device)
    new_img9, history9 = texture_hijacking_attack(model, img, target, mask9, orig_label=pred_orig_idx)

    plot_attack_curves([history3,history6, history7, history8, history9], [ "PGD","quantile", "quantileinverse", "random", "quantile_proto"])
    # plot_attack_comparison(img, new_img8, heatmap, mask8, None)
    # plot_attack_comparison(img, new_img6, heatmap, mask6, None)
    # plot_attack_comparison(img, new_img9, proto_heatmap, mask9, None)
    # from torchvision.utils import save_image
    # save_image(new_img3.cpu(), 'results/stage3/hijacked.png')

    # normalize_new_img = normalize(new_img)
    # out_orig = model(normalize_new_img)
    # pred_new_idx = out_orig.argmax(1).item()
    # heatmap = explainer.generate(normalize_new_img, pred_new_idx)
    # save_visual_result(normalize_new_img, heatmap, categories[pred_new_idx], categories[target], 0, T.Normalize(
    #         mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    #         std=[1/0.229, 1/0.224, 1/0.225]
    #     ))
pet_categories = None
def get_diagnose_dataloader():
    dataset = torchvision.datasets.OxfordIIITPet(
        root="./data", 
        download=True, 
        split="test",
        transform=T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor()
        ])
    )
    global pet_categories
    pet_categories = dataset.classes
    return dataset

def run_batch_experiment(model, dataset, explainer, target_idx, proto_heatmap, num_samples=100):
    stats = []
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, (img, label) in enumerate(tqdm(dataloader, desc="Batch Hijacking")):
        if i >= num_samples: break
        
        img = img.to(device) 
        # attack(model, categories, img, target_idx, explainer, proto_heatmap, True)
        
        with torch.no_grad():
            output = model(normalize(img))
            orig_pred = output.argmax(1).item()
        
        heatmap = explainer.generate(normalize(img), orig_pred)
        
        mask_quantile = get_mask(heatmap, method="quantile").to(device)
        mask_pgd = torch.ones_like(mask_quantile)
        mask_quantilei = get_mask(heatmap, method="quantileinverse").to(device)
        mask_proto = get_mask(proto_heatmap, method="quantile").to(device)
        mask_protoi = get_mask(proto_heatmap, method="quantileinverse").to(device)

        
        res_proto, hist_pr = texture_hijacking_attack(model, img, target_idx, mask_proto, orig_label=orig_pred)
        res_protoi, hist_pri = texture_hijacking_attack(model, img, target_idx, mask_protoi, orig_label=orig_pred)
        res_pgd, hist_p = texture_hijacking_attack(model, img, target_idx, mask_pgd, orig_label=orig_pred)
        res_quantile, hist_q = texture_hijacking_attack(model, img, target_idx, mask_quantile, orig_label=orig_pred)
        res_quantilei, hist_qi = texture_hijacking_attack(model, img, target_idx, mask_quantilei, orig_label=orig_pred)
        random_all = []
        mask_random = get_mask(heatmap, method="random").to(device)
        res_random, hist_r = texture_hijacking_attack(model, img, target_idx, mask_random, orig_label=orig_pred)

        # def show_heatmap(img_tensor):
        #     normalize_new_img = normalize(img_tensor)
        #     out_orig = model(normalize_new_img)
        #     pred_new_idx = out_orig.argmax(1).item()
        #     heatmap = explainer.generate(normalize_new_img, pred_new_idx)
        #     save_visual_result(normalize_new_img, heatmap, categories[pred_new_idx], categories[target_idx], 0, T.Normalize(
        #             mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        #             std=[1/0.229, 1/0.224, 1/0.225]
        #         ))
        # show_heatmap(res_quantile.detach())
        # show_heatmap(res_random.detach())
        
        def check_success(hist):
            return any(p >= 0.7 for p in hist['target_prob'])

        sample_stat = {
            'idx': i,
            'proto_success': check_success(hist_pr),
            'protoi_success': check_success(hist_pri),
            'pgd_success': check_success(hist_p),
            'quantile_success': check_success(hist_q),
            'quantilei_success': check_success(hist_qi),
            'quantiler_success': check_success(hist_r),
            'proto_steps': len(hist_pr['target_prob']), 
            'protoi_steps': len(hist_pri['target_prob']),
            'pgd_steps': len(hist_p['target_prob']),
            'quantile_steps': len(hist_q['target_prob']), 
            'quantilei_steps': len(hist_qi['target_prob']),
            'quantiler_steps': len(hist_r['target_prob']),
        }
        stats.append(sample_stat)
        
    return stats

def print_detailed_summary(stats):
    total = len(stats)
    
    method_map = {
        'proto': 'Prototype Top-10%',
        'protoi': 'Prototype Bottom-10%',
        'pgd': 'Full-PGD',
        'quantile': 'Origin Top-10%',
        'quantilei': 'Origin Bottom-10%',
        'quantiler': 'Random-10%'
    }

    print("\n" + "="*75)
    print(f"{'Attack Method':<20} | {'ASR (Success)':<15} | {'Avg Steps':<12} | {'Efficiency'}")
    print("-" * 75)

    for key, display_name in method_map.items():
        success_list = [s[f'{key}_success'] for s in stats]
        asr = sum(success_list) / total
        
        steps_list = [s[f'{key}_steps'] for s in stats if s[f'{key}_success']]
        avg_steps = np.mean(steps_list) if steps_list else 0
        
        efficiency = (asr * 100) / (avg_steps + 1e-5)

        print(f"{display_name:<20} | {asr:>14.2%} | {avg_steps:>11.1f} | {efficiency:>10.2f}")
    
    print("="*75)
    print(f"Total samples evaluated: {total}\n")

def main():
    model = resnet50(weights=weights).to(device)
    model.eval()

    explainer = GradCAM(model, "layer3.5")
    for name, _ in model.named_modules():
        print(name)
    args = get_args()
    preprocess = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
    ])
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    prototype_image, _ = get_prototype(args.target)
    prototype_image = preprocess(prototype_image).clone().to(device)
    prototype_image = normalize(prototype_image).unsqueeze(0)
    proto_heatmap = explainer.generate(prototype_image, args.target)
    print(args.path)
    if args.path != None:
        attack(model, categories, Image.open(args.path).convert('RGB'), args.target, explainer, proto_heatmap)
    else:
        dataset = get_diagnose_dataloader()
        stats = run_batch_experiment(model, dataset, explainer, args.target,proto_heatmap, 100)
        print_detailed_summary(stats)

    

    

if __name__ == '__main__':
    main()
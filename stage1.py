import random
import numpy as np
import torch
import cv2
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from models import ResNet
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from tqdm import tqdm
from explainers import CAM, GradCAM
import argparse
from utils import save_visual_result, plot_deletion_metric

CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_args():
    parser = argparse.ArgumentParser(description="Stage 1: Interpretability & Faithfulness")

    parser.add_argument('--method', type=str, default='gradcam', choices=['cam', 'gradcam'], help='Explain Method')

    parser.add_argument('--num_samples', type=int, default=5, help='Number of photos to visualize, -1 for all images')
    parser.add_argument('--indices', type=int, nargs='+', default=None, help='Choose specific photos, e.g. --indices 102 405 89')

    parser.add_argument('--vis_only', action='store_true', help='Only generate heatmap result')
    parser.add_argument('--failure_only', action='store_true', help='Only use photos that are not classified correctly')
    
    
    return parser.parse_args()

def deletion_metric(model, img, label, heatmap, step=100, method="blur", metric="prob"):
    height, width = img.shape[2:]
    total_pixels = int(width * height * 0.5)
    per_pixels = int(np.ceil(total_pixels / step))
    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    pixels_heat = heatmap_resized.flatten()
    indices = np.argsort(pixels_heat)[::-1]
    random_indices = np.random.permutation(indices)


    if method == "blur":
        blurred_img = transforms.functional.gaussian_blur(img, kernel_size=[11, 11], sigma=[15.0, 15.0])
        if metric == "logit":
            with torch.no_grad():
                initial_logit = model(img)[0, label].item()
                final_logit = model(blurred_img)[0, label].item()

        def one_run(del_indices):
            del_pixels = 0
            x = []
            y = []
            mask = torch.ones((height, width), device=device)
            with torch.no_grad():
                for i in range(step + 1):
                    x.append(del_pixels / total_pixels * 0.5)
                    input1 = img * mask + blurred_img * (1 - mask)
                    logits = model(input1)
                    if metric == "prob":
                        output = F.softmax(logits, dim=1)
                        prob1 = output[0, label].item()
                        y.append(prob1)
                    else:
                        y.append(logits[0, label].item())
                    start = i * per_pixels
                    if start >= total_pixels:
                        break
                    end = min((i + 1) * per_pixels, total_pixels)
                    for j in del_indices[start : end]:
                        r, c = divmod(j, width)
                        mask[r, c] = 0
                    del_pixels += end - start
            return x, y
    else:
        if metric == "logit":
            with torch.no_grad():
                initial_logit = model(img)[0, label].item()
                final_logit = model(torch.full((1, 3, height, width), -1, device=device).float())[0, label].item()
        def one_run(del_indices):
            del_pixels = 0
            x = []
            y = []
            temp_img = img.clone()
            with torch.no_grad():
                for i in range(step + 1):
                    x.append(del_pixels / total_pixels * 0.5)
                    logits = model(temp_img)
                    if metric == "prob":
                        output = F.softmax(logits, dim=1)
                        prob1 = output[0, label].item()
                        y.append(prob1)
                    else:
                        y.append(logits[0, label].item())
                    start = i * per_pixels
                    if start >= total_pixels:
                        break
                    end = min((i + 1) * per_pixels, total_pixels)
                    for j in del_indices[start : end]:
                        r, c = divmod(j, width)
                        temp_img[0, :, r, c] = 0
                    del_pixels += end - start
            return x, y
    
    x, y_del = one_run(indices)
    all_random = []
    for i in range(10):
        _, y = one_run(np.random.permutation(indices))
        all_random.append(y)
    
    y_random = np.mean(all_random, axis=0)

    y_del = np.array(y_del)
    y_random = np.array(y_random)
    y_del = (y_del - final_logit) / (initial_logit - final_logit)
    y_random = (y_random - final_logit) / (initial_logit - final_logit)
    # y_del = np.clip(y_del, 0, 1)
    # y_random = np.clip(y_random, 0, 1)

    auc = (np.trapezoid(y_del, x), np.trapezoid(y_random, x))
    return x, y_del, y_random, auc
    


def main():
    args = get_args()

    model = ResNet()
    state_dict = torch.load("./models/resnet.pth")
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.method == 'cam':
        explainer = CAM(model, target_layer_name="resstage3", fc_layer_name="fclinear.2")
    else:
        explainer = GradCAM(model, target_layer_name="resstage3")

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), 
                        (0.5, 0.5, 0.5)),
    ])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    if args.indices:
        subset = args.indices
    else:
        if args.num_samples >= 0:
            subset = random.sample(range(len(testset)), k=args.num_samples)
        else:
            subset = [i for i in range(len(testset))]
    sub_testset = Subset(testset, subset)
    testloader = DataLoader(sub_testset, batch_size=1, shuffle=False)

    auc_del = 0
    auc_random = 0

    for i, (img, label) in enumerate(tqdm(testloader, desc="stage1", leave=False)):
        img = img.to(device)
        label = label.to(device)
        outputs = model(img)
        _, preds = torch.max(outputs, 1)
        pred_idx = preds.item()
        true_idx = label.item()
        is_correct = (pred_idx == true_idx)
        if is_correct and args.failure_only:
            continue
        heatmap = explainer.generate(img, class_idx=pred_idx)
        save_visual_result(img[0], heatmap, CLASSES[pred_idx], CLASSES[true_idx], subset[i], transforms.Normalize([-1, -1, -1], [2, 2, 2]))
        if not args.vis_only:
            x, y1, y2, auc = deletion_metric(model, img, pred_idx, heatmap, metric="logit")
            auc_del += auc[0]
            auc_random += auc[1]
            plot_deletion_metric(x, y1, y2, auc, CLASSES[pred_idx])
        
    print(f"total_samples:{len(testloader)}")
    print(f"auc_del_avg:{auc_del / len(testloader):.3f}")
    print(f"auc_random_avg:{auc_random / len(testloader):.3f}")
            
if __name__ == '__main__':
    main()
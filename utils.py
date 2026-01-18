import cv2
import numpy as np
import os
import torch
import PIL
import torchvision
import matplotlib.pyplot as plt

def save_visual_result(img_tensor, heatmap, pred_label_name, true_label_name, index, transform):

    img = transform(img_tensor)
    img = img.squeeze().cpu().permute(1, 2, 0).numpy()
    img = np.uint8(np.clip(img * 255, 0, 255))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    height, width = img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = np.clip(heatmap_resized, 0, 1)
    
    heatmap_color = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_color, cv2.COLORMAP_JET)

    overlaid_img = cv2.addWeighted(img, 0.5, heatmap_color, 0.3, 0)

    combined_img = np.hstack((img, heatmap_color, overlaid_img))

    # if pred_label == true_label:
    #     save_path = os.path.join(output_dir, f"correct/index_{index}_pred_{CLASSES[pred_label]}_true_{CLASSES[true_label]}.png")
    # else:
    #     save_path = os.path.join(output_dir, f"wrong/index_{index}_pred_{CLASSES[pred_label]}_true_{CLASSES[true_label]}.png")
    # cv2.imshow(save_path, combined_img)
    # cv2.imwrite(save_path, combined_img)
    img_rgb = cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB) 
    plt.figure(figsize=(15, 5)) 
    plt.imshow(img_rgb, interpolation="bicubic")
    plt.title(f"Idx:{index} | Pred:{pred_label_name} | True:{true_label_name}")
    plt.axis('off')
    plt.show() 

def plot_deletion_metric(x, y_del, y_random, auc, label_name):
    plt.figure(figsize=(6, 6))
    plt.plot(x, y_del, color='#1f77b4', lw=2, label='Deletion Curve')
    plt.plot(x, y_random, label='Random Baseline', color='red', linestyle='--', linewidth=2)
    plt.fill_between(x, y_del, color='#1f77b4', alpha=0.3)
    
    plt.xlim(-0.02, 1.02)
    plt.ylim(min(0, np.min(y_del), np.min(y_random))-0.01, max(1, np.max(y_del), np.max(y_random))+0.02)
    plt.xlabel('Pixels removed', fontsize=14)
    plt.ylabel(f'L[{label_name}]', fontsize=14)
    
    plt.text(0.4, 0.9, f'AUC(Deletion)={auc[0]:.3f}\nAUC(Random)={auc[1]:.3f}', fontsize=16, fontweight='bold')
    
    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.tight_layout()
    plt.show()

def patch_shuffling_tensor(img_tensor, grid_size=8):
    c, h, w = img_tensor.shape
    patch_h, patch_w = h // grid_size, w // grid_size
    
    patches = []
    for i in range(grid_size):
        for j in range(grid_size):
            patches.append(img_tensor[:, i * patch_h:(i + 1) * patch_h, j * patch_w:(j + 1) * patch_w])
    
    import random
    random.shuffle(patches)
    rows = []
    for i in range(grid_size):
        rows.append(torch.cat(patches[i * grid_size : (i + 1) * grid_size], dim=2))
    shuffled_tensor = torch.cat(rows, dim=1)
    
    return shuffled_tensor

def plot_attack_comparison(orig_img, adv_img, heatmap, mask, save_path):
    def to_np(t):
        t = t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
        return np.clip(t, 0, 1)

    plt.figure(figsize=(20, 4))
    plt.subplot(1, 4, 1)
    plt.imshow(to_np(orig_img))
    plt.title("Original Image", fontsize=12)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(mask.detach().cpu().squeeze(), cmap='gray')
    plt.title("Guidance Mask", fontsize=12)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(to_np(adv_img))
    plt.title("Adversarial (Target)", fontsize=12)
    plt.axis('off')

    plt.subplot(1, 4, 4)
    diff = (adv_img - orig_img).abs()
    diff = diff / (diff.max() + 1e-8)
    plt.imshow(to_np(diff))
    plt.title("Perturbation (Amplified)", fontsize=12)
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_attack_curves(history_list, labels):
    
    plt.figure(figsize=(15, 6))
    
    colors = [
        '#E31A1C', # 鲜红 
        '#1F78B4', # 深蓝 
        '#33A02C', # 深绿
        '#FF7F00', # 橙色
        '#6A3D9A', # 深紫
        '#FB9A99', # 浅粉
        '#A6CEE3', # 浅蓝
        '#B2DF8A', # 浅绿
        '#FDBF6F', # 浅橙
        '#CAB2D6', # 浅紫
        '#FFFF99', # 柠檬黄
        '#B15928', # 棕色
        '#000000', # 纯黑
        '#808080'  # 灰色
    ]

    plt.subplot(1, 2, 1)
    for i, (hist, label) in enumerate(zip(history_list, labels)):
        color = colors[i % len(colors)]
        plt.plot(hist['target_logit'], color=color, label=f'{label} (Target)', linestyle='-', linewidth=2)
        plt.plot(hist['orig_logit'], color=color, label=f'{label} (Original)', linestyle='--', linewidth=1.5, alpha=0.7)
        
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Logit Score', fontsize=12)
    plt.title('Logit Trajectory: Target vs. Original', fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.subplot(1, 2, 2)
    for i, (hist, label) in enumerate(zip(history_list, labels)):
        color = colors[i % len(colors)]
        iters_done = len(hist['target_prob'])
        plt.plot(hist['target_prob'], color=color, label=f'{label} (Target)', linestyle='-', linewidth=2)
        plt.plot(hist['orig_prob'], color=color, label=f'{label} (Original)', linestyle='--', linewidth=1.5, alpha=0.7)
        plt.axvline(x=iters_done-1, color=color, linestyle=':', linewidth=1.5, alpha=0.8)
        plt.text(iters_done-1, plt.gca().get_ylim()[1]*0.9, f'{iters_done-1}', color=color, 
                 ha='center', va='bottom', fontsize=8, fontweight='bold', 
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    plt.axhline(y=0.7, color='black', linestyle='-.', alpha=0.3, label='Threshold') 
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.ylim(-0.05, 1.05)
    plt.title('Confidence Shift: Target vs. Original', fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

def visualize_mdetr_gradcam(original_image, heatmap, caption, answer, confidence):
    if isinstance(original_image, PIL.Image.Image):
        img = np.array(original_image)
    else:
        img = original_image.copy()
    
    height, width, _ = img.shape
    print(height, width)
    print(heatmap.shape)

    heatmap_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    heatmap_resized = heatmap_resized.clip(0, 1)

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    superimposed_img = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.3, 0)
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB) 

    plt.figure(figsize=(12, 8))
    
    plt.imshow(superimposed_img)
    plt.axis('off')
    title_text = f"Question: {caption}\nPredicted Answer: {answer} ({confidence:.2f}%)"
    plt.title(title_text, fontsize=14, color='white', backgroundcolor='black')

    plt.tight_layout()
    plt.show()

    return superimposed_img
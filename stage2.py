import torch
import os
import torchvision
import cv2
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import clip
from PIL import Image
import numpy as np
from tqdm import tqdm
from utils import patch_shuffling_tensor, save_visual_result, visualize_mdetr_gradcam
from explainers import GradCAM, CLIPGradCAM, MDETRGradCAM, TransformerGradCAM
import argparse
from MDETR import plot_inference_qa

import platform
import pathlib

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
pet_categories = None

def get_args():
    parser = argparse.ArgumentParser(description="Stage 2: Spatial Test")

    parser.add_argument('--run', type=str, default="shuffle", choices=['shuffle', 'clip', 'mdetr'], help='Run which test')
    parser.add_argument('--path', type=str, default=None, help='Image path')

    return parser.parse_args()

for folder in ["results/stage2/consistent", "results/stage2/inconsistent"]:
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_diagnose_dataloader(batch_size=32):
    dataset = torchvision.datasets.OxfordIIITPet(
        root="./data", 
        download=True, 
        split="test",
    )
    global pet_categories
    pet_categories = dataset.classes
    return dataset


def save_sample(orig_tensor, shuff_tensor, label_name, pred_name, folder, idx):
    # (ImageNet mean/std)
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    to_pil = T.ToPILImage()
    
    orig_img = to_pil(inv_normalize(orig_tensor[0]).clamp(0, 1))
    shuff_img = to_pil(inv_normalize(shuff_tensor[0]).clamp(0, 1))
    
    combined = Image.new('RGB', (448, 224))
    combined.paste(orig_img, (0, 0))
    combined.paste(shuff_img, (224, 0))
    
    filename = f"{folder}/sample_{idx}_{label_name}_to_{pred_name}.jpg"
    combined.save(filename)

def shuffle_test():
    weights = ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]
    model = resnet50(weights=weights).to(device)
    model.eval()

    explainer = GradCAM(model, "layer4.2")
    
    dataset = get_diagnose_dataloader()

    stats = {'correct_orig': 0, 'correct_shuff': 0, 'logit_gain': [], 'more': 0}
    
    base_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor()
    ])
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    count_c, count_i = 0, 0

    for i in tqdm(range(len(dataset))):
        img, _ = dataset[i]

        testimg = base_transform(img)

        orig_t = normalize(testimg).unsqueeze(0).to(device)
        out_orig = model(orig_t)
        pred_orig_idx = out_orig.argmax(1).item()
        logit_orig = out_orig[0, pred_orig_idx]

        label_name = categories[pred_orig_idx].split(',')[0]
        heatmap = explainer.generate(orig_t, pred_orig_idx)
        save_visual_result(orig_t, heatmap, label_name, pet_categories[_], i, T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ))

        shuff_raw = patch_shuffling_tensor(testimg, grid_size=4)
        shuff_t = normalize(shuff_raw).unsqueeze(0).to(device)
        out_shuff = model(shuff_t)
        pred_shuff_idx = out_shuff.argmax(1).item()
        logit_shuff = out_shuff[0, pred_shuff_idx]
        pred_name = categories[pred_shuff_idx].split(',')[0]
        heatmap = explainer.generate(shuff_t, pred_shuff_idx)
        save_visual_result(shuff_t, heatmap, pred_name, pet_categories[_], i, T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ))
        print(logit_orig, logit_shuff)

        if pred_orig_idx == pred_shuff_idx:
            stats['correct_shuff'] += 1
            if logit_shuff > logit_orig:
                stats['more'] += 1
            stats['logit_gain'].append((logit_shuff - logit_orig).item())

        stats['correct_orig'] += 1


    print("\n" + "="*50)
    print(f"Results remain same: {len(stats['logit_gain'])}/{len(dataset)}")
    print(f"Results gain more confidence: {stats['more']/len(stats['logit_gain']):.2%}")
    print(f"Average Logit change: {np.mean(stats['logit_gain']):+.2f}")
    print("="*50)

def CLIPtest(image_path="images/apple_red_green.jpg"):
    print("Loading CLIP RN50...")
    model, _ = clip.load("RN50", device=device)
    model.eval()

    explainer = CLIPGradCAM(model, "visual.layer4")

    test_cases = [
        {
            "img_path": image_path, 
            "queries": [
                "the color of the left apple is red",      # True Query
                "the color of the left apple is green",     # False Query (Spatial Failure Test)
                "the left apple",
                "the right apple",

            ]
        }
    ]

    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    for case in test_cases:
        if not os.path.exists(case["img_path"]):
            print(f"Skipping: {case['img_path']} not found.")
            continue

        raw_img = Image.open(case["img_path"]).convert('RGB')
        img_t = preprocess(raw_img).unsqueeze(0).to(device)

        print(f"\nAnalyzing image: {case['img_path']}")
        
        heatmaps = []
        logits = []

        for query_text in case["queries"]:
            text_tokens = clip.tokenize([query_text]).to(device)
            
            with torch.no_grad():
                l_per_img, _ = model(img_t, text_tokens)
                logit_val = l_per_img[0, 0].item()
                logits.append(logit_val)

            heatmap = explainer.generate(img_t, text_tokens)
            heatmaps.append(heatmap)
            
            print(f"Query: '{query_text}' | Logit: {logit_val:.2f}")

        save_clip_results(img_t, heatmaps, case["queries"], logits, case["img_path"])

def save_clip_results(img_tensor, heatmaps, queries, logits, img_path):
    inv_normalize = T.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    img_np = inv_normalize(img_tensor[0]).cpu().permute(1, 2, 0).numpy()
    img_np = np.uint8(np.clip(img_np * 255, 0, 255))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    results = []
    height, width, _ = img_np.shape
    for i, (hm, q, l) in enumerate(zip(heatmaps, queries, logits)):
        hm = hm.astype(np.float32)
        hm = cv2.resize(hm, (width, height), interpolation=cv2.INTER_CUBIC)
        hm_uint8 = np.uint8(255 * np.clip(hm, 0, 1))
        hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_bgr, 0.5, hm_color, 0.3, 0)
        
        cv2.putText(overlay, f"Logit: {l:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        results.append(overlay)

    combined = np.hstack(results)
    save_path = f"results/stage2/clip_diagnosis_{os.path.basename(img_path)}"
    cv2.imwrite(save_path, combined)
    print(f"Saved CLIP diagnosis to {save_path}")

def MDETRtest(img_path="images/mdetr.png"):
    question = "What is on the table?"
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model_qa = torch.hub.load('mdetr-main', 'mdetr_efficientnetB5_gqa', source="local", pretrained=False, return_postprocessor=False)
    checkpoint_path = "models/gqa_EB5_checkpoint.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
    model_qa.load_state_dict(checkpoint["model"], strict=False) 
    model_qa = model_qa.cuda()
    model_qa.eval()
    import json
    with open("data/gqa_answer2id_by_type.json", "r", encoding='utf-8') as f:
        answer2id_by_type = json.load(f)
    id2answerbytype = {}                                                       
    for ans_type in answer2id_by_type.keys():                        
        curr_reversed_dict = {v: k for k, v in answer2id_by_type[ans_type].items()}
        id2answerbytype[ans_type] = curr_reversed_dict 
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).cuda()
    explainer = TransformerGradCAM(model_qa, "transformer.encoder.layers.5", "backbone.0.body.blocks.6.2")
    explainer = MDETRGradCAM(model_qa, "backbone.0.body.blocks.6.2")
    for name, _ in model_qa.named_modules():
        print(name)
    memory_cache = model_qa(img_tensor, [question], encode_and_save=True)
    outputs = model_qa(img_tensor, [question], encode_and_save=False, memory_cache=memory_cache)
    type_conf, type_pred = outputs["pred_answer_type"].softmax(-1).max(-1)
    ans_type_str = ["obj", "attr", "rel", "global", "cat"][type_pred.item()]
    ans_conf, ans_idx_tensor = outputs[f"pred_answer_{ans_type_str}"][0].softmax(-1).max(-1)
    ans_idx = ans_idx_tensor.item()
    # ans_idx = 24
    # ans_conf = outputs[f"pred_answer_{ans_type_str}"][0].softmax(-1)[ans_idx]
    answer_text = id2answerbytype[f"answer_{ans_type_str}"][ans_idx]
    print(answer_text)
    total_conf = type_conf.item() * ans_conf.item() * 100
    heatmap = explainer.generate(img_tensor, question, ans_type_str, ans_idx)
    visualize_mdetr_gradcam(img, heatmap, question, answer_text, total_conf)
    plot_inference_qa(model_qa, img, question, id2answerbytype)




if __name__ == "__main__":
    args = get_args()
    if args.run == "shuffle":
        shuffle_test()
    if args.run == "clip":
        if args.path != None:
            CLIPtest(args.path)
        else:
            CLIPtest()
    if args.run == "mdetr":
        if args.path != None:
            MDETRtest(args.path)
        else:
            MDETRtest()

# Can’t See the Forest for the Trees: Diagnosing the Spatial Blindness in CNN-based Models via Grad-CAM
---

This repository contains the code for my final project.


## 1. Installation

The code has been tested on Windows 11 with a single NVIDIA GPU (≥ 8GB memory).


- Python 3.12.8
- PyTorch 2.9.1
- CUDA 12.6 (optional, CPU execution is supported for visualization)


1. Install [Conda](https://www.anaconda.com/) and create a `Conda` environment.

```bash
conda create -n gradcam python=3.12.8
conda activate gradcam
```

2. Install PyTorch-2.9.1 with pip3 according to the official documentation.

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

3. Clone this repository and install the requirements.

```bash
git clone https://github.com/VVKAHPM/2025-Fall-Computer-Vision-Final-Project.git
cd  2025-Fall-Computer-Vision-Final-Project
pip install -r requirements.txt
```

4. Clone the repository of the prototype images for stage3.

```bash
git clone https://github.com/EliSchwartz/imagenet-sample-images.git
```

5. Download the [checkpoint](https://zenodo.org/record/4721981/files/gqa_EB5_checkpoint.pth?download=1) and put it in `./models/`

```plaintext
YourProject/
├── models/
│   └── gqa_EB5_checkpoint.pth  <-- put it here
│   └── resnet.pth
│   └── ...
├── ...
```

## 2. Stage 1

This stage include basic CAM and Grad-CAM implementation and Deletion metric.

Use `python stage1.py -h` for more help.

CIFAR-10 Dataset will be downloaded into `./data` at the first time you run `stage1.py`.

If you don't want to see visulization result, please comment out the call of `save_visual_result`, `deletion_metric` and `plot_deletion_metric` functions.

## 3. Stage 2

This stage include patch shuffle test, CLIP test and MDETR test.

### 3.1 Patch shuffle

To run patch shuffle test, you should use command `python stage2.py --run shuffle` 

Oxford-IIIT Pet Dataset will be downloaded into `./data` at the first time you run `stage.py --run shuffle`

If you don't want to see visulization result, please comment out the call of `save_visual_result`

### 3.2 CLIP

To run CLIP test, you should use command `python stage2.py --run clip` 

CLIP model will be downloaded at the first time you run `stage.py --run clip`. You can modify the code in `CLIPtest` to change the way to load CLIP-ResNet-50.

You can modify the `queries` attribute in `test_cases` in the function `CLIPtest` to test different queries.

You can use `--path [PATH]` to specify an image you want to use as an input. The default is `./images/apple_red_green.jpg`

### 3.3 MDETR

To run MDETR test, you should use command `python stage2.py --run mdetr` 

You can modify the `question` in the first line of the function `MDETRtest` to test different questions.

You can use `--path [PATH]` to specify an image you want to use as an input. The default is `./images/mdetr.png`

## 4. Stage 3

This stage include adversarial attack experiment.

To run this experiment, you should use command `python stage3.py` 

You can use `--path [PATH]` to specify an image you want to attack. The default is `None`. If the path is `None`, the program will test on the first 100 images of Oxford-IIIT Pet Dataset and give a result of different mask methods.

You can use `--target [NUMBER]` to specify a class as the attack target. The number is the [ImageNet](https://www.image-net.org/) Dataset class number. The default is `42`, the class index of `agama`.

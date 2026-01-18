# Can’t See the Forest for the Trees: Diagnosing the Spatial Blindness in CNN-based Models via Grad-CAM
---

This repository contains the code for my final project.


## 1. Installation

The code has been tested on Windows 11 with a single NVIDIA GPU (≥ 8GB memory).
All experiments were conducted using the exact dependency versions specified in
`requirements.txt`.

### Environment (other versions may also be compatible)

- Python 3.12.8
- PyTorch 2.9.1
- CUDA 12.6 (optional, CPU execution is supported for visualization)

### Create environment

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

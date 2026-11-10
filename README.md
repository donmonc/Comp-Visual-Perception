# Comp-Visual-Perception

**Computational Visual Perception**

## Project Overview

This project investigates how different computer vision systems process and interpret **Mooney images** in comparison with their original image counterparts.

Mooney images are high-contrast, typically black-and-white images in which substantial visual detail is removed while some underlying structure may remain perceptible.

### Overall Project Goal

- How well do state-of-the-art vision systems understand Mooney images?
- How do biologically constrained vision systems perceive Mooney images?

We investigate these questions by passing the same set of original and Mooney-transformed images through four different computer vision systems, each representing a different aspect of visual processing.

## Vision Systems

### 1. DINOv2 — Visual Representation Learning

**Purpose:** Extract general-purpose visual representations from images.

DINOv2 is a self-supervised vision foundation model that learns visual features without requiring manually annotated labels. It produces numerical image embeddings that can be used for downstream computer vision tasks.

**Our approach:**
- Extract embeddings from original and Mooney images.
- Compare corresponding image embeddings using cosine similarity.
- Analyse the distribution of similarity scores.
- Investigate the most and least similar image pairs.

**Selected model:** `facebook/dinov2-base`

**Original paper:** [DINOv2: Learning Robust Visual Features without Supervision (2023)](https://arxiv.org/abs/2304.07193)

### 2. Qwen2.5-VL — Vision-Language Understanding

**Purpose:** Investigate how a vision-language model interprets visual information.

Qwen2.5-VL is a multimodal model capable of processing images and generating natural-language responses. It supports visual recognition, object localisation, image understanding and reasoning.

**Our approach:**
- Provide original and Mooney images to the model using consistent prompts.
- Compare generated descriptions and responses.
- Examine which visual attributes and facial characteristics the model can identify in each image type.

**Selected model:** `Qwen/Qwen2.5-VL-3B-Instruct`

**Original technical report:** [Qwen2.5-VL Technical Report (2025)](https://arxiv.org/abs/2502.13923)

### 3. Depth Anything V2 — Monocular Depth Estimation

**Purpose:** Investigate how visual transformations affect depth estimation.

Depth Anything V2 is a monocular depth estimation model that predicts relative depth information from a single image.

**Our approach:**
- Generate depth maps for original and Mooney images.
- Compare the resulting depth predictions.
- Examine whether the loss of visual information in Mooney images affects the model's depth estimates.

**Selected model:** `depth-anything/Depth-Anything-V2-Small-hf`

**Original paper:** [Depth Anything V2 (2024)](https://arxiv.org/abs/2406.09414)

### 4. BiSeNet — Facial Semantic Segmentation

**Purpose:** Investigate how facial components are identified and segmented in original and Mooney images.

BiSeNet (Bilateral Segmentation Network) is a semantic segmentation architecture. We use a face-parsing implementation trained to segment facial regions into distinct semantic classes.

**Our approach:**
- Generate facial segmentation masks for original and Mooney images.
- Examine the segmentation of regions such as skin, eyes, eyebrows, nose, mouth, hair and background.
- Compare the resulting masks and investigate which facial structures remain distinguishable after Mooney transformation.

**Selected implementation:** [BiSeNet Face Parsing](https://github.com/yakhyo/face-parsing)

**Original architecture paper:** [BiSeNet: Bilateral Segmentation Network for Real-Time Semantic Segmentation (ECCV 2018)](https://arxiv.org/abs/1808.00897)

**Related dataset paper:** [MaskGAN: Towards Diverse and Interactive Facial Image Manipulation (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Lee_MaskGAN_Towards_Diverse_and_Interactive_Facial_Image_Manipulation_CVPR_2020_paper.html) — introduces CelebAMask-HQ, a facial parsing dataset used to train the selected implementation.

## Dataset and Experimental Setup

All four systems are evaluated using the same image collection:

- **Original:** 100 original face images.
- **Mooney:** 100 corresponding Mooney-transformed images.
- **Subset:** 10 selected original images and their corresponding Mooney versions for focused analysis.

The objective is to maintain consistent inputs across all four systems to enable meaningful comparisons.

## Results

Each vision system has a dedicated results directory:

| Directory | Vision system |
|---|---|
| `Result_1/` | DINOv2 |
| `Result_2/` | Qwen2.5-VL |
| `Result_3/` | Depth Anything V2 |
| `Result_4/` | BiSeNet |

Results include model outputs for both original and Mooney images, alongside relevant quantitative comparisons and visualisations.


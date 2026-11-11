## Result 4: BiSeNet — Facial Semantic Segmentation

### Objective

Investigate how Mooney transformations affect the ability of a pretrained facial parsing system to identify and segment individual facial components.

We use a pretrained **BiSeNet ResNet18 face-parsing model**, trained using CelebAMask-HQ, to produce 19-class segmentation masks for all 100 original images and their corresponding Mooney images.

**Selected implementation:** [BiSeNet Face Parsing — yakhyo/face-parsing](https://github.com/yakhyo/face-parsing)

**Original architecture paper:** [BiSeNet: Bilateral Segmentation Network for Real-time Semantic Segmentation (ECCV 2018)](https://arxiv.org/abs/1808.00897)

**Related dataset paper:** [MaskGAN: Towards Diverse and Interactive Facial Image Manipulation (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Lee_MaskGAN_Towards_Diverse_and_Interactive_Facial_Image_Manipulation_CVPR_2020_paper.html)

### Existing Outputs

The inference pipeline has successfully processed all 100 image pairs.

```text
Result_4/
├── original/
│   ├── masks/
│   ├── class_masks/
│   ├── color_masks/
│   └── overlays/
├── mooney/
│   ├── masks/
│   ├── class_masks/
│   ├── color_masks/
│   └── overlays/
└── bisenet_comparison.csv
```

The model predicts 19 semantic classes, including background, skin, eyebrows, eyes, ears, nose, mouth, lips, neck, clothing, hair and accessories.

### Proposed Analysis

#### 1. Facial Class Coverage

Calculate the percentage of pixels assigned to each semantic class for every image.

Compare the average and median class coverage between original and Mooney images.

Pay particular attention to:
- Skin.
- Left and right eyebrows.
- Left and right eyes.
- Ears.
- Nose.
- Mouth and lips.
- Hair.

**Purpose:** Examine which facial components occupy a substantial proportion of the model's predictions under each image condition.

#### 2. Class Detection Frequency

Calculate how frequently each facial class is predicted across the 100 images under both conditions.

**Purpose:** Determine which facial components BiSeNet identifies consistently in original images and whether those classes remain detectable in Mooney images.

#### 3. Segmentation Class Diversity

Compare the number of distinct semantic classes predicted per image.

Report:
- Mean number of classes detected.
- Median number of classes detected.
- Distribution of class counts across both conditions.

**Purpose:** Investigate whether Mooney transformations reduce the diversity of semantic facial predictions.

#### 4. Segmentation Consistency

Compare corresponding original and Mooney masks using:

- Pixel-level class agreement.
- Foreground mask Intersection over Union (IoU).
- Per-class IoU and Dice scores.

**Purpose:** Examine the spatial and semantic consistency of the predicted segmentation masks across the two image conditions.

#### 5. Class Coverage Difference

Calculate the difference in predicted pixel coverage for each facial class between original and Mooney images.

Visualise changes in class coverage across all 100 pairs.

**Purpose:** Identify which facial regions experience the largest changes in predicted segmentation.

#### 6. Qualitative Segmentation Comparisons

Select representative image pairs displaying:

- Detailed facial segmentation in the original but limited segmentation in Mooney.
- Facial components that remain distinguishable in both conditions.
- Incorrect or unexpected class assignments in Mooney images.

Display original images, Mooney images, colour-coded masks and overlays side by side.

### Proposed Output Structure

```text
Result_4/
├── original/
├── mooney/
├── analysis/
│   ├── summary_statistics.json
│   ├── class_coverage_comparison.csv
│   └── class_detection_frequency.csv
├── visualizations/
│   ├── average_class_coverage.png
│   ├── class_detection_frequency.png
│   ├── segmentation_class_diversity.png
│   ├── pixel_agreement_distribution.png
│   └── qualitative_comparisons.png
└── bisenet_comparison.csv
```

### Interpretation Considerations

- The model was trained on conventional facial images, so Mooney images represent a substantially different visual input.
- The predicted segmentation masks are model outputs, not ground-truth annotations.
- A class assigned to a pixel does not necessarily mean that the corresponding facial component is genuinely present.
- Low pixel agreement indicates differences in predictions, not necessarily poor segmentation accuracy.
- Foreground IoU measures overlap of non-background regions; it does not measure whether the correct facial classes were assigned.

### Expected Deliverable

A comparative analysis of facial semantic segmentation in original and Mooney images, supported by class coverage statistics, class detection frequency, segmentation consistency metrics and qualitative mask visualisations.

## Result 3: Depth Anything V2 — Monocular Depth Estimation

### Objective

Investigate how transforming original face images into Mooney images affects the relative-depth structures predicted by a monocular depth estimation model.

We use the pretrained **Depth Anything V2 Small** model to generate depth maps for all 100 original images and their 100 corresponding Mooney images.

**Selected model:** `depth-anything/Depth-Anything-V2-Small-hf`

**Original paper:** [Depth Anything V2 (2024)](https://arxiv.org/abs/2406.09414)

### Existing Outputs

The inference pipeline has successfully processed all 100 image pairs.

```text
Result_3/
├── original/
│   ├── face_000_idx60.npy
│   ├── face_000_idx60.png
│   └── ...
├── mooney/
│   ├── face_000_idx60.npy
│   ├── face_000_idx60.png
│   └── ...
└── depth_comparison.csv
```

Each image has:
- A raw floating-point depth prediction (`.npy`).
- A grayscale depth visualisation (`.png`).

### Proposed Analysis

#### 1. Overall Depth Statistics

Calculate descriptive statistics across the dataset for both image conditions:

- Mean predicted depth.
- Median predicted depth.
- Standard deviation.
- Minimum and maximum predicted values.
- Percentile distributions.

**Purpose:** Examine differences in the numerical distributions of the model's predictions.

These statistics should be treated as descriptive only because the model predicts relative depth, not calibrated physical distances.

#### 2. Depth Map Visual Comparison

Display original images, Mooney images and their corresponding predicted depth maps side by side.

Generate representative contact sheets for visual inspection.

**Purpose:** Investigate whether depth structures and spatial variations remain distinguishable after Mooney transformation.

#### 3. Spatial Depth Structure Consistency

Compare corresponding original and Mooney depth maps using:

- Spearman rank correlation.
- Structural Similarity Index (SSIM).
- Normalised pixel-wise differences.

Normalise depth maps appropriately before structural comparisons, since independently generated relative-depth predictions may have different numerical scales.

**Purpose:** Examine whether the spatial organisation of predicted depth is retained across the two image conditions.

#### 4. Depth Map Difference Visualisation

Generate difference maps between corresponding original and Mooney depth predictions after consistent per-image normalisation.

**Purpose:** Highlight regions where the predicted spatial depth structure differs most between the two conditions.

#### 5. Highest and Lowest Similarity Examples

Identify representative image pairs with the highest and lowest spatial depth agreement.

Display the original image, Mooney image, both predicted depth maps and their difference map.

### Proposed Output Structure

```text
Result_3/
├── original/
├── mooney/
├── analysis/
│   ├── summary_statistics.json
│   └── depth_similarity_results.csv
├── visualizations/
│   ├── depth_statistics.png
│   ├── depth_similarity_distribution.png
│   ├── top_similarity_examples.png
│   ├── bottom_similarity_examples.png
│   └── depth_difference_examples.png
└── depth_comparison.csv
```

### Interpretation Considerations

- Depth Anything V2 Small is a relative-depth estimation model, not a metric-depth model.
- Predicted values should not be interpreted as distances in metres.
- Raw mean and standard deviation differences do not necessarily indicate changes in actual facial depth.
- Structural comparisons should account for differences in prediction scale.
- Depth-map similarity is not equivalent to depth-estimation accuracy.

### Expected Deliverable

A comparative analysis of predicted depth structures in original and Mooney images, supported by depth visualisations, descriptive statistics, spatial similarity metrics and representative image pairs.

---
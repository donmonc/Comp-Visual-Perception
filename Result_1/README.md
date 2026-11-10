## Result 1: DINOv2 — Original vs. Mooney Representation Analysis

### Objective

Investigate how transforming original face images into Mooney images affects the visual representations extracted by DINOv2.

We use the pretrained **DINOv2 Base (`facebook/dinov2-base`)** model to extract 768-dimensional image embeddings from all 100 original images and their 100 corresponding Mooney images.

The main comparison metric is **cosine similarity** between the embedding of each original image and its corresponding Mooney image.

### Existing outputs

The inference pipeline has already processed all 100 image pairs. The outputs are available in:

```text
Result_1/
├── original/                  # 100 original image embeddings (.npy)
├── mooney/                    # 100 Mooney image embeddings (.npy)
└── dinov2_comparison.csv      # Paired cosine similarity results
```

The `.npy` files contain the extracted embeddings, while the CSV contains the similarity score and associated metadata for each pair.

**Important:** The analysis should use these existing outputs. There is no need to rerun DINOv2 inference.

### Proposed Analysis

#### 1. Overall Similarity Statistics

Calculate descriptive statistics across the 100 matched image pairs:

- Number of image pairs.
- Mean cosine similarity.
- Median cosine similarity.
- Standard deviation.
- Minimum and maximum similarity.
- 25th and 75th percentiles.
- Interquartile range (IQR).
- Number and proportion of pairs with negative cosine similarity.

**Purpose:** Establish an overall picture of how strongly DINOv2's representations align between original images and their Mooney transformations.

#### 2. Similarity Distribution

Generate a histogram of the 100 cosine similarity scores.

Optionally include a density curve or mark the mean and median.

**Purpose:** Examine the distribution of similarities, identify clusters or concentration ranges, and understand the variability across image pairs.

#### 3. Highest and Lowest Similarity Cases (Top 3 and Bottom 3)

Identify:

- Top 3 image pairs with the highest cosine similarity.
- Bottom 3 image pairs with the lowest cosine similarity.

For each pair, display the original image, its corresponding Mooney image and the cosine similarity score.

Generate two comparison contact sheets:

- Top 3 highest-similarity pairs.
- Bottom 3 lowest-similarity pairs.

**Purpose:** Visually investigate which image characteristics may be associated with stronger or weaker representation similarity.

#### 4. Ranked Similarity Plot

Generate a sorted plot of all 100 image pairs, ranked from lowest to highest cosine similarity.

A horizontal bar chart or dot plot is recommended.

**Purpose:** Show the variation across the complete dataset and make outliers and differences between image pairs easy to identify.

#### 5. Cross-Image Retrieval (Recommended Additional Experiment)

Investigate whether a Mooney image can retrieve its corresponding original image based on DINOv2 embedding similarity.

For each Mooney embedding:

1. Calculate cosine similarity against all 100 original image embeddings.
2. Rank the original images by similarity.
3. Check whether the corresponding original image appears at rank 1 or within the top 5.

Report:

- Top-1 retrieval accuracy.
- Top-5 retrieval accuracy.
- Mean Reciprocal Rank (MRR).

**Purpose:** Evaluate whether DINOv2 preserves enough visual information to associate Mooney images with their corresponding original images, relative to other images in the dataset.

This is a cross-image representation retrieval experiment, not a facial identity-recognition experiment.

#### 6. Embedding Visualisation (Optional)

Use dimensionality reduction techniques such as PCA, and optionally UMAP, to project the 768-dimensional embeddings into two dimensions.

Plot:

- Original image embeddings.
- Mooney image embeddings.
- Corresponding original–Mooney pairs, optionally connected by lines.

**Purpose:** Explore whether original and Mooney images form distinguishable clusters in DINOv2's representation space and how far corresponding pairs are separated.

### Proposed Output Structure

```text
Result_1/
├── original/
├── mooney/
├── analysis/
│   ├── summary_statistics.json
│   ├── top_3_pairs.csv
│   ├── bottom_3_pairs.csv
│   ├── retrieval_results.csv
│   └── retrieval_summary.json
├── visualizations/
│   ├── similarity_histogram.png
│   ├── ranked_similarity.png
│   ├── top_3_comparison.png
│   ├── bottom_3_comparison.png
│   └── embedding_pca.png
└── dinov2_comparison.csv
```

### Interpretation and Reporting Considerations

- Higher cosine similarity indicates greater alignment between two image embeddings in DINOv2's feature space.
- Lower similarity indicates less alignment, but does not necessarily mean that the model has failed.
- Similarity scores should not be interpreted as probabilities or direct measures of facial identity preservation.
- Visual inspection of the highest- and lowest-similarity pairs should accompany the numerical analysis.
- All comparisons must use correctly matched image filenames.
- Record the model identifier (`facebook/dinov2-base`), embedding dimension (768) and other relevant experiment metadata.

### Expected Deliverable

A concise analysis of how Mooney transformations affect DINOv2's visual representations, supported by descriptive statistics, plots, selected image comparisons and, if implemented, cross-image retrieval results.

The analysis should provide both quantitative and visual evidence that can be used in the project presentation.
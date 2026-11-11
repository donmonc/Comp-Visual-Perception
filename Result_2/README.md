## Result 2: Qwen2.5-VL — Vision-Language Understanding

### Objective

Investigate how **Qwen2.5-VL-3B-Instruct** interprets original face images and their corresponding Mooney transformations.

The model was given the same prompt for both conditions, without being told whether an image was original or Mooney. It generated descriptions for **100 original images and 100 Mooney images**.

- **Model:** `Qwen/Qwen2.5-VL-3B-Instruct`
- **Technical report:** [Qwen2.5-VL Technical Report (2025)](https://arxiv.org/abs/2502.13923)
- **Results:** `Result_2/qwen_comparison.csv`, `Result_2/original/` and `Result_2/mooney/`

### Analysis Tasks

The following analyses should be conducted using the existing results. **No model regeneration is required.**

#### 1. Facial Feature Recognition

Compare how Qwen describes these facial regions across the two conditions:

- Face outline
- Eyes
- Eyebrows
- Nose
- Mouth/lips
- Ears

For each feature, examine whether it is:
- Clearly distinguishable
- Ambiguous or uncertain
- Not clearly discernible
- Not mentioned

Calculate the number and percentage of responses mentioning each feature for both conditions. Where possible, distinguish a simple mention from a claim that the feature is clearly visible.

**Output:** A comparison table and grouped bar chart showing feature-mention frequencies for original and Mooney images.

#### 2. Description and Response Comparison

Compare the descriptions generated for each corresponding image pair. Examine:

- Number of facial features described.
- Level of visual detail and specificity.
- Spatial relationships between facial regions.
- Whether the model expresses uncertainty.
- Differences between what it describes in the original and Mooney versions.

Look for cases where the model describes a feature confidently in one condition but struggles to describe it in the other.

**Output:** A concise summary of the main differences, supported by selected paired examples.

#### 3. Response Length and Uncertainty

Compare response word counts and uncertainty expressions across both conditions.

Initial results show:
- Mean response length: **71.0 words (original)** vs. **98.1 words (Mooney)**.
- Mooney responses are longer in 72 pairs, original responses are longer in 27, and one pair has equal lengths.

Investigate whether longer Mooney descriptions reflect additional visual detail, repeated observations, attempts to explain ambiguity, or unsupported speculation. Do not interpret verbosity as better visual understanding.

**Output:** A response-length comparison chart and a short interpretation of the uncertainty patterns.

#### 4. Qualitative Error Analysis

Manually inspect selected original–Mooney pairs and their corresponding model descriptions.

Prioritise examples where:
- The original receives a detailed description, but the Mooney version does not.
- Some facial structures remain identifiable in both versions.
- The model claims to see features that are difficult to distinguish visually.
- The model gives ambiguous, contradictory or potentially unsupported descriptions.

Pay particular attention to the distinction between a feature that is genuinely distinguishable and one whose presence is merely inferred from the overall face-like pattern.

**Output:** Side-by-side image pairs with shortened excerpts from both model responses and a brief explanation of each example.

### Suggested Report Slides

**Slide 1 — Quantitative Comparison**
- Grouped bar chart of facial feature mentions in original vs. Mooney images.
- Response-length comparison.
- Key numerical observations and a short interpretation.

**Slide 2 — Qualitative Comparison**
- Two or three representative original–Mooney image pairs.
- Corresponding model descriptions or selected excerpts.
- Brief annotations explaining what the model identifies, misses or misinterprets.

**Slide 3 — Key Findings and Limitations** (if space permits)
- Main differences in feature recognition and descriptive behaviour.
- Examples of uncertainty, unsupported claims or hallucinations.
- What the results suggest about the model's interpretation of Mooney images.

### Interpretation Considerations

- Model-generated descriptions are not ground-truth visual annotations.
- A feature not mentioned does not necessarily mean it is visually absent.
- Distinguish feature mentions from actual visual distinguishability.
- Manually review claims that appear unsupported or overconfident.
- Response length alone is not evidence of better visual understanding.
- One Mooney response (`face_020_idx57`) appears to end abruptly and may be truncated. Flag it as a data-quality limitation rather than treating it as a normal complete response.
- Interpret the results as an exploratory comparison of model behaviour, not a definitive measure of human or machine face perception.

### Expected Deliverable

A concise, evidence-based comparison of Qwen's descriptions of original and Mooney images, supported by quantitative feature-mention statistics, response comparisons, qualitative examples and key limitations.
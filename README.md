# Image Denoising using Regularization Methods

This repository implements and evaluates five core regularization-based methods for image denoising:

1. Tikhonov (L2) Regularization
2. Total Variation (TV)
3. Wavelet-based Approaches
4. Non-Local Means (NLM)
5. Weighted Nuclear Norm Minimization (WNNM)

## Overview

Image denoising is a fundamental challenge in image processing, affecting diverse domains such as medical diagnostics, remote sensing, surveillance, and photography. This project explores classical regularization methods with particular emphasis on their mathematical formulations, including primal, dual, and KKT conditions, to showcase how each method balances noise suppression against structure preservation.

## Problem Statement

Image denoising is frequently posed as a convex optimization problem in which we aim to find a balance between fitting observed (noisy) data and enforcing prior knowledge about the true image structure. A general form is:

```
min (1/2)||y - x||²₂ + λ R(x)
 x
```

where:
- y denotes the noisy observation
- x is the denoised image
- λ is a regularization weight
- R(·) is a regularizer reflecting assumptions (e.g., smoothness, sparsity, or low-rank structure)

## Implemented Methods

### 1. Tikhonov (L2) Regularization

**Primal Formulation:**
```
min (1/2)||x - y||²₂ + (λ/2)||Lx||²₂
 x
```

**Characteristics:**
- Simplicity, closed-form or fast FFT-based solutions
- Minimal parameter tuning (just λ)
- Computational Complexity: O(N) or O(N log N) with FFT
- Weakness: Over-smoothing of edges and textures

### 2. Total Variation (TV) Minimization

**Primal Formulation:**
```
min (1/2)||x - y||²₂ + λ||∇x||₁
 x
```

**Characteristics:**
- Effective edge preservation due to L¹ gradient norm
- Convex formulation with guaranteed global optimum
- Computational Complexity: O(N·K) for K iterations
- Weakness: Staircasing effect in smooth regions

### 3. Wavelet-Based Denoising

**Primal Formulation:**
```
min (1/2)||x - y||²₂ + λ∑|αₖ|
 x                   k
```

**Characteristics:**
- Multi-scale representation and computational efficiency
- Theoretical near-optimality in certain noise models
- Computational Complexity: O(N)
- Weakness: Basis selection can introduce artifacts

### 4. Non-Local Means (NLM)

**Primal Formulation:**
```
min (1/2)∑(xᵢ - yᵢ)² + (λ/2)∑wᵢⱼ(xᵢ - xⱼ)²
 x      i                i,j
```

**Characteristics:**
- Excellent detail and texture preservation
- Robust to high noise levels if similar patches can be found
- Computational Complexity: O(N·S²·P²) for S×S search window and P×P patches
- Weakness: High computational cost

### 5. Weighted Nuclear Norm Minimization (WNNM)

**Primal Formulation:**
```
min (1/2)||X - Y||²ᶠ + λ∑wᵢσᵢ(X)
 X                    i
```

**Characteristics:**
- State-of-the-art performance among classical methods
- Effectively recovers fine structures
- Highest computational cost among the methods
- Weakness: Non-convex optimization can require careful initialization

## Experimental Results

Experiments were conducted on three grayscale images—Landscape, Peacock, and Spray—each degraded by additive Gaussian noise with standard deviations σ ∈ {0.01, 0.025, 0.05}.

### PSNR Comparison (in dB)

| Method  | Landscape |        |        | Peacock  |        |        | Spray    |        |        |
|---------|-----------|--------|--------|----------|--------|--------|----------|--------|--------|
|         | 0.01      | 0.025  | 0.05   | 0.01     | 0.025  | 0.05   | 0.01     | 0.025  | 0.05   |
| quad    | 16.88     | 15.42  | 16.13  | 20.77    | 17.63  | 18.42  | 17.25    | 14.20  | 15.99  |
| TV      | 14.92     | 14.32  | 14.81  | 21.16    | 18.18  | 18.30  | 16.62    | 15.09  | 14.59  |
| tik     | 19.91     | 16.60  | 17.94  | 16.98    | 12.64  | 13.38  | 19.51    | 14.84  | 16.75  |
| nlm     | 15.99     | 15.03  | 16.49  | 21.58    | 17.92  | 18.88  | 19.05    | 13.94  | 16.03  |
| wnnm    | 17.86     | 18.55  | 17.33  | 23.11    | 18.42  | 19.08  | 20.17    | 15.60  | 16.86  |
| wavelet | 19.65     | 18.20  | 19.99  | 20.83    | 15.38  | 15.38  | 20.96    | 17.15  | 18.73  |

### Key Observations

- **Landscape Image**: Tikhonov regularization achieves the highest PSNR at low noise levels, while wavelet-based denoising performs best as noise increases.
- **Peacock Image**: WNNM outperforms all other methods across all noise levels.
- **Spray Image**: Wavelet method achieves the highest PSNR, followed closely by WNNM.

## Conclusion

The results highlight the trade-offs between different regularization strategies:
- Smoothness-based methods (Tikhonov) work well for simple structures but struggle with high-frequency details
- Patch-based methods (WNNM) are more effective for textured images
- Wavelet-based denoising provides a good balance, adapting well across different noise levels

## Dependencies

- Python 3.6+
- NumPy
- SciPy
- OpenCV
- matplotlib (for visualization)

## Installation

```bash
git clone https://github.com/username/image-denoising.git
cd image-denoising
pip install -r requirements.txt
```

## Usage

```python
from denoising import tikhonov, tv, wavelet, nlm, wnnm

# Load a noisy image
noisy_image = cv2.imread('noisy_image.png', 0)  # 0 for grayscale

# Apply different denoising methods
tikhonov_result = tikhonov(noisy_image, lambda_param=0.1)
tv_result = tv(noisy_image, lambda_param=0.2)
wavelet_result = wavelet(noisy_image, lambda_param=0.3)
nlm_result = nlm(noisy_image, search_window=21, patch_size=7)
wnnm_result = wnnm(noisy_image, patch_size=8, group_size=16)

# Calculate PSNR (if ground truth is available)
from metrics import calculate_psnr
psnr = calculate_psnr(ground_truth, denoised_image)
```

## References

1. Tikhonov, A.N. (1963). Solution of incorrectly formulated problems and the regularization method. Soviet Math. Dokl., 4:1035–1038.
2. Rudin, L.I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. Physica D: Nonlinear Phenomena, 60(1-4):259–268.
3. Donoho, D.L. & Johnstone, I.M. (1994). Ideal spatial adaptation by wavelet shrinkage. Biometrika, 81(3):425–455.
4. Buades, A., Coll, B., & Morel, J.M. (2005). A non-local algorithm for image denoising. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), volume 2, pages 60–65. IEEE.
5. Gu, S., Zhang, L., Zuo, W., & Feng, X. (2014). Weighted nuclear norm minimization with application to image denoising. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2862–2869.

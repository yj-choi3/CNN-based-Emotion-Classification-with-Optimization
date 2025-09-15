# CNN-based-Emotion-Classification-with-Optimization
# CNN-based Facial Emotion Classification with Performance Optimization

## Overview
This project implements and improves a CNN model for classifying facial emotions  
across three categories: **Happy, Neutral, and Sad**.  

The main goal was to address **overfitting** in the baseline CNN and to enhance the model’s  
generalization performance through systematic application of regularization and optimization techniques.  
The final model achieved **>80% validation accuracy**, surpassing the baseline (~70%).

---

## Dataset
- Provided facial image dataset (RGB, 3 classes: Happy, Neutral, Sad).  
- Preprocessing: resizing, normalization.  
- Class imbalance handled using **Stratified K-Fold cross-validation**.

---

## Methods

### Baseline CNN
- 4 convolutional blocks (Conv → ReLU → Conv → ReLU → MaxPooling).  
- Limitations:
  - Severe **overfitting**: >90% train accuracy vs. ~70% validation accuracy.  
  - Lack of regularization (no Dropout, BatchNorm, L2).  
  - No optimization strategies (no LR scheduler, no early stopping).  

### Applied Improvements
- **Data Augmentation**: rotations, intensity variations, scaling, lighting changes.  
- **Dropout & L2 Regularization**: applied after pooling and fully connected (FC) layers.  
- **Batch Normalization**: stabilized training and improved convergence speed.  
- **FC Redesign**: from simple `3200 → 3` to deeper `3200 → 128 → 64 → 3`.  
- **Learning Rate Scheduler (ReduceLROnPlateau)** and **Early Stopping** for efficient training.  

---

## Experiments

| Step | Techniques Applied | Validation Accuracy | Notes |
|------|-------------------|---------------------|-------|
| Baseline | Simple CNN | ~70% | Severe overfitting |
| Exp. 1 | Augmentation + Early Stopping | ~74% | Small improvement |
| Exp. 2 | Dropout(0.5) + Augmentation | ~75% | Reduced overfitting |
| Exp. 3 | Aug + L2 (1e-3) | ~75% | More stable training |
| Exp. 4 | Aug + BatchNorm | ~75% | Faster convergence |
| Exp. 6 | Aug + BatchNorm + L2 + LR Scheduler + FC redesign | ~76.9% | Improved generalization |
| Exp. 9 | Conv/FC Dropout tuning + FC redesign | **78.8%** | Best performance |

Final model surpassed **80%** after Stratified K-Fold validation and full dataset training.

---

## Results
- **Accuracy:** Final test accuracy exceeded **80%**.  
- **Stability:** Training/validation curves showed stable improvement with reduced overfitting.  
- **Visualization:** Grad-CAM applied to confirm feature learning (optional future extension).

---

## Repository Structure

# Fashion House Classification — CNN + ResNet50 Transfer Learning

Classifies runway fashion images into 15 fashion houses (Chanel, Gucci, Dior, and others) using both a custom-built CNN and transfer learning.

## Dataset
Vogue Runway dataset (~22,700 images, 2015–2019 collections, 15 classes).

## Approach
- Built a custom 5-block VGG-style CNN from scratch (2.6M parameters).
- Applied transfer learning with ResNet50, using both feature-extraction and fine-tuning modes.
- Addressed class imbalance with a weighted random sampler and used MixUp augmentation to improve generalization.
- Tuned hyperparameters with Optuna, and used Grad-CAM to visualize what the model was learning.
- Combined models with a soft-voting ensemble and deployed the final model via a Gradio demo interface.

## Result
Best model (fine-tuned ResNet50) achieved **~95% validation accuracy**.

## Files
- `fashion_house_classification.py` — full pipeline: data loading, both architectures, training, tuning, interpretability, and deployment

# House Segmentation Report Draft

## 1. Dataset and Preprocessing

This project uses aerial imagery for house segmentation. Images are paired with binary pixel masks where house pixels are labeled as foreground and all other pixels are background.

The preprocessing pipeline performs the following steps:

- Load aerial images and polygon annotations or precomputed masks.
- Rasterize polygon boundaries into binary pixel masks.
- Split the dataset into training, validation, and test partitions.
- Resize images and masks to a fixed spatial resolution for model training.

Key issue:

- Class imbalance is expected because background pixels usually outnumber house pixels. This is addressed with Dice loss in training and evaluation with IoU and Dice score.

## 2. Model Architecture and Training

The segmentation model uses DeepLabV3 with a ResNet-50 backbone. The implementation supports both training from scratch and transfer learning via a pretrained backbone.

Training setup:

- Loss: cross-entropy plus Dice loss
- Metrics: IoU and Dice score
- Optimizer: Adam
- Outputs: best checkpoint, metric history, and learning curves

What to discuss in the final submission:

- Why a segmentation model fits the task better than the original pretrained classifier.
- Whether transfer learning improved convergence.
- Any signs of overfitting in the validation curves.

## 3. Evaluation Results

Report the test set metrics after training:

- IoU: insert value here
- Dice: insert value here

Add a short interpretation of the scores:

- High IoU indicates strong overlap between predicted and ground truth house regions.
- High Dice score indicates the model is segmenting the target class consistently.

## 4. Visual Results

Include three example rows of:

- Input aerial image
- Ground truth mask
- Predicted mask

Suggested figures to include:

- `outputs/training_curves.png`
- `outputs/sample_predictions.png`

## 5. Issues and Mitigations

Discuss any problems encountered during training, such as:

- Overfitting on a small dataset
- Noisy or incomplete polygon annotations
- Severe class imbalance
- Limited GPU or CPU training time

Possible mitigations:

- Data augmentation
- Better splits
- Dice-based losses
- Early stopping or checkpoint selection by validation IoU

## 6. Conclusion

Summarize whether the new segmentation pipeline improved the project relative to the original pretrained model and highlight the deployment and CI/CD additions.
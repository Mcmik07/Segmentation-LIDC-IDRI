# Segmentation-LIDC-IDRI
This repository contains the second stage of my Lung Cancer project. For the first stage, please refer to my LIDC-IDRI-Preprocessing repository.

1) LIDC-IDRI-Preprocessing (Stage 1)

The LIDC-IDRI-Preprocessing repository covers the preprocessing pipeline for the LIDC-IDRI dataset. After running the scripts/notebook, you will obtain .npy files for each CT slice and its corresponding mask. In addition, the workflow generates meta.csv and clean_meta.csv after executing the Jupyter notebook.
2. Training:
Train the model. There will be more than 5 cases. More than five training cases are available, including both base and augmentation-based settings. Examples include UNET_base, UNET_with_augmentation, NestedUNET_base, and NestedUNET_with_augmentation.
If you want to use the augmented version, set augmentation=True.

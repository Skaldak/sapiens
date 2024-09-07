# Finetuning Sapiens: Surface Normal Estimation
This guide outlines the process to finetune the pretrained Sapiens model for surface normal estimation on custom data.


## 📂 1. Data Preparation
Set `$DATA_ROOT` as your training data root directory.\
We provide a toy dataset for easy start at `$DATA_ROOT=/uca_transient_a/open_source_from_manifold/rawalk/sapiens_finetune/task_normal`

The train data directory structure is as follows:

      $DATA_ROOT/
      ├── images/
      │   └── 00000000.png
      │   └── 00000001.png
      │   └── 00000002.png
      ├── masks/
      │   └── 00000000.png
      │   └── 00000001.png
      │   └── 00000002.png
      ├── normals/
      │   └── 00000000.npy
      │   └── 00000001.npy
      │   └── 00000002.npy

The folders as follows:\
-`$DATA_ROOT/images`: RGB images (.png or .jpg or .jpeg).\
-`$DATA_ROOT/mask`: Boolean masks for human pixels (.png, .jpg, or .jpeg). \
-`$DATA_ROOT/normals`: Ground truth surface normals (axis order: (X, Y, Z) or (Z, Y, X)).

## ⚙️ 2. Configuration Update

Edit `$SAPIENS_ROOT/seg/configs/sapiens_normal/normal_general/sapiens_1b_normal_general-1024x768.py`:

1. Set `pretrained_checkpoint` to your checkpoint path.
2. Update `dataset_train.data_root` to your `$DATA_ROOT`.
3. Modify `train_pipeline.RandomBackground.background_images_root` to a directory with non-human images for background augmentation. Optionally, you can remove this augmentation.
4. (Optional) Adjust hyperparameters like `num_epochs` and `optim_wrapper.optimizer.lr`.


## 🏋️ 3. Finetuning

The training scripts for Sapiens-1B are under: `$SAPIENS_ROOT/seg/scripts/finetune/normal_general/sapiens_1b`/
Make sure you have activated the sapiens python conda environment.

### A. 🚀 Single-node Training
For datasets < 100k samples, use `$SAPIENS_ROOT/seg/scripts/finetune/normal_general/sapiens_1b/node.sh`.

Key variables:
- `DEVICES`: GPU IDs (e.g., "0,1,2,3,4,5,6,7")
- `TRAIN_BATCH_SIZE_PER_GPU`: Default 2 for A100 GPU
- `OUTPUT_DIR`: Checkpoint and log directory
- `RESUME_FROM`: Checkpoint to resume training from. Starts training from previous epoch. Defaults to empty string.
- `LOAD_FROM`: Checkpoint to load weight from. Starts training from epoch 0. Defaults to empty string.
- `mode=multi-gpu`: Launch multi-gpu training with multiple workers for dataloading.
- `mode=debug`: (Optional) To debug. Launched single gpu dry run, with single worker for dataloading. Supports interactive debugging with pdb/ipdb.

Note, if you wish to finetune from an existing normal estimation checkpoint, set the `LOAD_FROM` variable.

Launch:
```bash
cd $SAPIENS_ROOT/seg/scripts/finetune/normal_general/sapiens_1b
./node.sh
  ```

### B. 🌐 Multi-node Training (Slurm)

Use `$SAPIENS_ROOT/seg/scripts/finetune/normal_general/sapiens_1b/slurm.sh`

Additional variables:
- `CONDA_ENV`: Path to conda environment
- `NUM_NODES`: Number of nodes (default 4, 8 GPUs per node)

Launch:
```bash
cd $SAPIENS_ROOT/seg/scripts/finetune/normal_general/sapiens_1b
./slurm.sh
  ```

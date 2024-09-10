import glob
import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torchvision
from mmseg.apis import inference_model, init_model

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

torchvision.disable_beta_transforms_warning()


def infer_mask(seqs):
    if world_size > 0:
        torch.cuda.set_device(local_rank)
    model = init_model(args.seg_config, args.seg_checkpoint, device="cuda")
    for seq in seqs:
        input_dir = os.path.abspath(seq)
        output_dir = input_dir.replace("align_images", "seg")
        os.makedirs(output_dir, exist_ok=True)
        print(local_rank, input_dir, output_dir)
        files = sorted(glob.glob(os.path.join(input_dir, "*")))
        for file in files:
            image = cv2.imread(file)  # has to be bgr image
            result = inference_model(model, image)
            seg = result.pred_sem_seg.data.cpu().numpy()[0] * 255
            cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + ".jpg"), seg)
    del model
    torch.cuda.empty_cache()


def infer_depth(seqs):
    if world_size > 0:
        torch.cuda.set_device(local_rank)
    model = init_model(args.depth_config, args.depth_checkpoint, device="cuda")
    for seq in seqs:
        input_dir = os.path.abspath(seq)
        mask_dir = input_dir.replace("align_images", "mask")
        output_dir = input_dir.replace("align_images", "depth")
        os.makedirs(output_dir, exist_ok=True)
        print(local_rank, input_dir, output_dir)
        files = sorted(glob.glob(os.path.join(input_dir, "*")))
        for file in files:
            image = cv2.imread(file)  # has to be bgr image
            mask = cv2.imread(os.path.join(mask_dir, os.path.splitext(os.path.basename(file))[0] + ".jpg"))
            result = inference_model(model, image)
            depth = result.pred_depth_map.data.cpu().numpy()[0].clip(0)
            depth[mask.max(2) == 0] = 1024
            cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + ".exr"), depth)
    del model
    torch.cuda.empty_cache()


def infer_normal(seqs):
    if world_size > 0:
        torch.cuda.set_device(local_rank)
    model = init_model(args.normal_config, args.normal_checkpoint, device="cuda")
    for seq in seqs:
        input_dir = os.path.abspath(seq)
        mask_dir = input_dir.replace("align_images", "mask")
        output_dir = input_dir.replace("align_images", "normal")
        os.makedirs(output_dir, exist_ok=True)
        print(local_rank, input_dir, output_dir)
        files = sorted(glob.glob(os.path.join(input_dir, "*")))
        for file in files:
            image = cv2.imread(file)  # has to be bgr image
            mask = cv2.imread(os.path.join(mask_dir, os.path.splitext(os.path.basename(file))[0] + ".jpg"))
            result = inference_model(model, image)
            normal = result.pred_depth_map.data.cpu().numpy().transpose(1, 2, 0)[..., ::-1]
            normal = normal / (np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8)
            normal[mask.max(2) == 0] = 0
            cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + ".exr"), normal)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--seg_config",
        help="Config file",
        default="seg/configs/sapiens_seg/goliath/sapiens_1b_goliath-1024x768.py",
    )
    parser.add_argument(
        "--seg_checkpoint",
        help="Checkpoint file",
        default="pretrained/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth",
    )
    parser.add_argument(
        "--depth_config",
        help="Config file",
        default="seg/configs/sapiens_depth/render_people/sapiens_2b_render_people-1024x768.py",
    )
    parser.add_argument(
        "--depth_checkpoint",
        help="Checkpoint file",
        default="pretrained/sapiens_2b_render_people_epoch_25.pth",
    )
    parser.add_argument(
        "--normal_config",
        help="Config file",
        default="seg/configs/sapiens_normal/normal_render_people/sapiens_2b_normal_render_people-1024x768.py",
    )
    parser.add_argument(
        "--normal_checkpoint",
        help="Checkpoint file",
        default="pretrained/sapiens_2b_normal_render_people_epoch_70.pth",
    )
    parser.add_argument("--input", help="Input image dir")
    parser.add_argument("--output", default=None, help="Path to output dir")
    parser.add_argument("--task", default="23", help="Task to run")
    args = parser.parse_args()

    seqs = sorted(filter(lambda x: os.path.isdir(x), glob.glob(args.input)))
    world_size = int(os.environ.get("WORLD_SIZE", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))

    if world_size > 0:
        seqs = [seq for idx, seq in enumerate(seqs) if local_rank < 0 or idx % world_size == local_rank]
    print(local_rank, len(seqs), "sequences")
    if "1" in args.task:
        infer_mask(seqs)
    if "2" in args.task:
        infer_depth(seqs)
    if "3" in args.task:
        infer_normal(seqs)
    print("done")

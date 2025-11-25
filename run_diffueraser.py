import torch
import os
import time
import argparse
import cv2
import numpy as np
from diffueraser.diffueraser import DiffuEraser
from propainter.inference import Propainter, get_device

def generate_mask_video(input_video_path, mask_coords, output_mask_path):
    """
    Generates a mask video based on relative coordinates.
    mask_coords: [left, top, right, bottom] (0-1 relative to video size)
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    left_rel, top_rel, right_rel, bottom_rel = mask_coords

    # Convert relative coordinates to pixels
    x1 = int(left_rel * width)
    y1 = int(top_rel * height)
    x2 = int(right_rel * width)
    y2 = int(bottom_rel * height)

    # Validate coordinates
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))

    print(f"Generating mask with box: ({x1}, {y1}) to ({x2}, {y2}) for {frame_count} frames.")

    # Create mask frame (single frame is enough since it's static)
    mask_frame = np.zeros((height, width, 3), dtype=np.uint8)
    # White rectangle on black background
    cv2.rectangle(mask_frame, (x1, y1), (x2, y2), (255, 255, 255), -1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_mask_path, fourcc, fps, (width, height))

    for _ in range(frame_count):
        out.write(mask_frame)

    cap.release()
    out.release()
    print(f"Mask video saved to: {output_mask_path}")

def main():

    ## input params
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_video', type=str, default="examples/example3/video.mp4", help='Path to the input video')
    parser.add_argument('--input_mask', type=str, default="examples/example3/mask.mp4" , help='Path to the input mask')
    parser.add_argument('--auto_mask', type=float, nargs=4, help='Auto generate mask with relative coordinates: left top right bottom (e.g., 0.1 0.1 0.5 0.5)')
    parser.add_argument('--video_length', type=int, default=10, help='The maximum length of output video')
    parser.add_argument('--mask_dilation_iter', type=int, default=8, help='Adjust it to change the degree of mask expansion')
    parser.add_argument('--max_img_size', type=int, default=960, help='The maximum length of output width and height')
    parser.add_argument('--save_path', type=str, default="results" , help='Path to the output')
    parser.add_argument('--ref_stride', type=int, default=10, help='Propainter params')
    parser.add_argument('--neighbor_length', type=int, default=10, help='Propainter params')
    parser.add_argument('--subvideo_length', type=int, default=50, help='Propainter params')
    parser.add_argument('--base_model_path', type=str, default="weights/stable-diffusion-v1-5" , help='Path to sd1.5 base model')
    parser.add_argument('--vae_path', type=str, default="weights/sd-vae-ft-mse" , help='Path to vae')
    parser.add_argument('--diffueraser_path', type=str, default="weights/diffuEraser" , help='Path to DiffuEraser')
    parser.add_argument('--propainter_model_dir', type=str, default="weights/propainter" , help='Path to priori model')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.auto_mask:
        mask_output_path = os.path.join(args.save_path, "auto_mask.mp4")
        generate_mask_video(args.input_video, args.auto_mask, mask_output_path)
        args.input_mask = mask_output_path

    priori_path = os.path.join(args.save_path, "priori.mp4")
    output_path = os.path.join(args.save_path, "diffueraser_result.mp4")

    ## model initialization
    device = get_device()
    # PCM params
    ckpt = "2-Step"
    video_inpainting_sd = DiffuEraser(device, args.base_model_path, args.vae_path, args.diffueraser_path, ckpt=ckpt)
    propainter = Propainter(args.propainter_model_dir, device=device)

    start_time = time.time()

    ## priori
    propainter.forward(args.input_video, args.input_mask, priori_path, video_length=args.video_length,
                        ref_stride=args.ref_stride, neighbor_length=args.neighbor_length, subvideo_length = args.subvideo_length,
                        mask_dilation = args.mask_dilation_iter)

    ## diffueraser
    guidance_scale = None    # The default value is 0.
    video_inpainting_sd.forward(args.input_video, args.input_mask, priori_path, output_path,
                                max_img_size = args.max_img_size, video_length=args.video_length, mask_dilation_iter=args.mask_dilation_iter,
                                guidance_scale=guidance_scale)

    end_time = time.time()
    inference_time = end_time - start_time
    print(f"DiffuEraser inference time: {inference_time:.4f} s")

    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()

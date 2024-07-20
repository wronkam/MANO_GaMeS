import argparse
import gc
import json
import os
import shutil

import torch
from tqdm import tqdm
import subprocess
import sys
from glob import glob

import torchvision

from utils.general_utils import PILtoTorch

from PIL import Image


def process_images(img_frame,save_path,images_path,masks_path):
    shutil.rmtree(save_path)  # clean whole dir, in particular colmap stuff
    os.makedirs(save_path, exist_ok=True)

    image_dest = os.path.join(save_path, 'input')
    os.makedirs(image_dest, exist_ok=True)

    image_mask_dest = os.path.join(save_path, 'images_masked')
    os.makedirs(image_mask_dest, exist_ok=True)

    cameras = glob(os.path.join(images_path, 'cam*'))
    masks = glob(os.path.join(masks_path, 'cam*'))
    images_count = 0
    for mask, cam in zip(sorted(masks), sorted(cameras)):
        camera_name = cam.split(os.sep)[-1]
        mask_path = os.path.join(mask, f'image{img_frame}.png')
        img_path = os.path.join(cam, f'image{img_frame}.jpg')
        if os.path.exists(mask_path) and os.path.exists(img_path):
            mask_img = Image.open(mask_path)
            mask_img = PILtoTorch(mask_img, mask_img.size)
            img = Image.open(img_path)
            img = PILtoTorch(img, img.size)

            masked_img = img * mask_img

            torchvision.utils.save_image(masked_img, os.path.join(image_mask_dest, f'{camera_name}.jpg'))
            shutil.copy(img_path, os.path.join(image_dest, f'{camera_name}.jpg'))
            images_count +=1
    return images_count


def extract_mano(args_capture,args_frame,args_hand,save_path, annotations):
    my_annot = annotations[args_capture][args_frame][args_hand]
    my_annot['is_rhand'] = True if args_hand == 'right' else False
    tmp = {}
    count = {}
    for frame in sorted(annotations[args_capture].keys()):
        if annotations[args_capture][frame][args_hand] is None:
            continue
        # json circular reference error
        tmp[frame] = {k: v for k, v in annotations[args_capture][frame][args_hand].items()}
        for k, v in annotations[args_capture][frame][args_hand].items():
            if k not in ['shape', 'pose']:
                continue
            if k not in count:
                count[k] = {}
            if len(v) not in count[k]:
                count[k][len(v)] = 0
            count[k][len(v)] += 1
    my_annot['all_frames'] = tmp
    with open(os.path.join(save_path, 'mano_params.json'), 'w') as f_out:
        json.dump(my_annot, f_out)


if __name__ == '__main__':
    """
    python .\\utils\interhands_frame_extractor.py  --annot-path .\data\InterHand\5\annotations\test\InterHand2.6M_test_MANO_NeuralAnnot.json /
    --capture 0  --hand right --save-path .\data\hands --images-path .\data\InterHand\5\InterHand2.6M_5fps_batch1\images\test\Capture0\ROM04_RT_Occlusion\ /
    --masks-path .\data\InterHand\5\InterHand2.6M_5fps_batch1\masks_removeblack\test\Capture0\ROM04_RT_Occlusion\ /
    --frame 18052 --try-all-frames --convert-path .\convert.py /
    --colmap-path ..\COLMAP\COLMAP-3.9.1-windows-cuda\COLMAP.bat 
    """
    parser = argparse.ArgumentParser(description="Interhands extraction parameters")
    parser.add_argument("--annot-path", type=str, required=True, help="Path to interhands MANO annotations json")
    parser.add_argument("--capture", type=str, required=True, help="interhands capture")
    parser.add_argument("--frame", type=str, required=True, help="interhands frame")
    parser.add_argument("--hand", type=str, required=True, help="interhands hand: left or right")
    parser.add_argument("--save-path", type=str, required=True, help="where to save extracted frame for colmap")
    parser.add_argument("--images-path", type=str, required=True,
                        help="where to find images (points to dir with per camera dirs)")
    parser.add_argument("--masks-path", type=str, required=True,
                        help="where to images masks (points to dir with per camera dirs)")
    parser.add_argument("--try-all-frames",  action='store_true', default=False,
                        help="ignores frame, tries all and picks the one with most colmap matches")
    parser.add_argument("--colmap-path", type=str, required=False,
                        help="path to colmap executable for convert.py")
    parser.add_argument("--convert-path", type=str, required=False, default=None,
                        help="path to convert.py from the main dir")
    parser.add_argument("--skip-to", type=str, required=False, default='-1',
                        help="skip frames before given")

    args, _ = parser.parse_known_args(sys.argv[1:])

    with open(args.annot_path,'r') as f_in:
        mano_annotations = json.load(f_in)
    if args.try_all_frames:
        pkl_path = os.path.join('colmap_frame_scores.pkl')
        if os.path.exists(pkl_path):
            scores = torch.load(pkl_path)
        else:
            scores = {}
        all_frames = sorted(mano_annotations[args.capture].keys())
        all_frames = [frame for frame in all_frames if int(frame) >= int(args.skip_to)]
        assert args.convert_path is not None
        colmap = "" if args.colmap_path is None else f' --colmap_executable {args.colmap_path}'
        bar = tqdm(all_frames,desc='Frames')
        for frame in bar:
            if frame in scores.keys():
                bar.write(f'Frame {frame} has score {scores[frame]} already recorded')
                continue
            count = process_images(frame, args.save_path, args.images_path, args.masks_path)
            bar.write(f'Trying frame: {frame}. {count} images found')
            if count <= 0:
                continue
            extract_mano(args.capture, frame, args.hand, args.save_path, mano_annotations)
            cmd = f'python {args.convert_path} -s {args.save_path}{colmap}'
            subprocess.run(cmd.split(' '), capture_output=True)  # suppresses outputs
            # os.system(cmd)
            scores[frame] = len(glob(os.path.join(args.save_path, 'images','cam*')))
            torch.save(scores,pkl_path)
            bar.write(f'Frame: {frame}. It got {scores[frame]} images found by Colmap. '
                      f'The best was {max(scores.values())}')
            gc.collect()
        max_score = max(scores.values())
        potentials = [key for key in scores.keys() if scores[key] >= max_score]
        print('Max score:', max_score)
        if len(potentials) > 1:
            print('Go manually through the following frames, choose one in which hand is in '
                  'the most flat position (fingers least bent, furthers from the fist)')
            print(potentials)
            print("Rerun the program with no --try-all-frames and provide the chosen frame number")
        else:
            frame = potentials[0]
            print(f"{frame} is the best with {max_score} images, out of {potentials}. Processing it")
            extract_mano(args.capture, args.frame, args.hand, args.save_path, mano_annotations)
            process_images(args.frame, args.save_path, args.images_path, args.masks_path)
            cmd = f'python {args.convert_path} -s {args.save_path}{colmap}'
            subprocess.run(cmd.split(' '), capture_output=True)  # suppresses outputs
            # os.system(cmd)
        print('Done!')

    else:
        extract_mano(args.capture,args.frame,args.hand,args.save_path, mano_annotations)
        process_images(args.frame,args.save_path,args.images_path,args.masks_path)

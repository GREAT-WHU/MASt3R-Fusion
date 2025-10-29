import torch
import yaml
import pickle 
from mast3r_fusion.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
from mast3r_fusion.multiprocess_utils import new_queue, try_get_msg
from mast3r_fusion.config import load_config, config, set_global_config
from mast3r_fusion.dataloader import Intrinsics, load_dataset
from mast3r_fusion.frame import Mode, SharedKeyframes, SharedStates, create_frame
import numpy as np
import h5py
import io
import cv2
from mast3r_fusion.geometry import (
    constrain_points_to_ray,
)
import matplotlib.pyplot as plt
from natsort import natsorted
import tqdm
import time
import argparse

def len_h5(h5_filename):
    with h5py.File(h5_filename, "r") as f:
        return len(f.keys())

def mask_sky(img):
    mm = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (mm > 250).astype(np.uint8)
    h, w = mask.shape
    flood_mask = np.zeros((h+2, w+2), np.uint8)
    out_mask = np.zeros_like(mask, dtype=np.uint8)
    for x in range(w):
        if mask[0, x] == 1:
            cv2.floodFill(mask, flood_mask, (x, 0), 2)
    out_mask = (mask == 2)
    return out_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mast3r_fusion visualization with options")

    parser.add_argument("--frame_id", type=int, default=240)
    parser.add_argument("--h5", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument("--calib", type=str)
    parser.add_argument("--pose_file", type=str, default =None)

    args = parser.parse_args()

    FRAME_ID = args.frame_id

    load_config(args.config)
    config["use_calib"] = True

    manager = mp.Manager()
    main2viz = new_queue(manager, False)
    viz2main = new_queue(manager, False)

    def load_frame_from_h5(h5_filename, iframe):
        with h5py.File(h5_filename, "r") as f:
            blob = bytes(f[f"frame_{iframe}"][()])
            buffer = io.BytesIO(blob)
            return torch.load(buffer, map_location="cpu")

    dataset = load_dataset("")
    dataset.subsample(2,0,999999)

    if config["use_calib"]:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
            False, intrinsics.get("model","pinhole"), intrinsics.get("scale",1), intrinsics.get("height_new",None)
        )

    device = 'cuda'
    H5_FILE = args.h5

    id_poses = {}

    if not args.pose_file is None:
        pppp = np.loadtxt(args.pose_file)
        for i in range(len(pppp)):
            id_poses[int(pppp[i,15])] = pppp[i,1:9]
    else:
        print('Scanning poses...')
        for i in tqdm.tqdm(range(len_h5(args.h5))):
            dd = load_frame_from_h5(H5_FILE, i)
            id_poses[i] = dd['T_WC'][-1,:]
            h,w = dd['uimg'].shape[:2]


    keyframes = SharedKeyframes(manager, h, w,buffer=1024)
    states = SharedStates(manager, h, w)
    K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
                device, dtype=torch.float32
            )
    keyframes.set_intrinsics(K)



    for i in tqdm.tqdm(range(len_h5(args.h5))):
        is_nearby=False
        for ii in range(FRAME_ID-10,FRAME_ID+10):
            if np.linalg.norm(id_poses[i][0:3] - id_poses[ii][0:3])<30: 
                is_nearby = True
        if not is_nearby:continue
        dd = load_frame_from_h5(H5_FILE, i)
        dd['T_WC'][-1,:] = torch.tensor(id_poses[i])
        dd['X'] *= dd['T_WC'][-1,-1]
        dd['X'] = dd['X'][None]
        dd['T_WC'][-1,-1] = 1.0

        dd['X'][0,dd['X'][0,:,2]>15.0,:] = 0.01
        frame = create_frame(i, dd['uimg'].astype(np.float32)/255.0, dd['T_WC'], img_size=dataset.img_size, device=device)
        frame.update_pointmap(dd['X'], dd['C']/dd['N'])
        frame.feat = 0
        frame.pos = 0
        frame.dataset_idx = i
        states.set_frame(frame)
        keyframes.append(frame)
        states.set_mode(Mode.TRACKING)

    run_visualization(config, states, keyframes, main2viz, viz2main, max_show = 1000)





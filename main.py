import argparse
import datetime
import pathlib
import sys
import time
import cv2
import lietorch
import torch
import tqdm
import yaml
from mast3r_fusion.global_opt import FactorGraph

from mast3r_fusion.config import load_config, config, set_global_config
from mast3r_fusion.dataloader import Intrinsics, load_dataset
import mast3r_fusion.evaluate as eval
from mast3r_fusion.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_fusion.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_fusion.multiprocess_utils import new_queue, try_get_msg
from mast3r_fusion.tracker import FrameTracker
from mast3r_fusion.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp
import numpy as np
from scipy.spatial.transform import Rotation

import pickle
import io
import h5py

def find_valid_numbers(a, b):
    result = []
    for i, c in enumerate(b):
        if abs(c - a) <= 1:
            continue 
        close_indices = [j for j, d in enumerate(b) if abs(d - c) <= 20]
        if i == min(close_indices) or c == a - 2 :
            result.append(c)
    return result


def run_backend(states, keyframes):
    mode = states.get_mode()
    if mode == Mode.INIT or states.is_paused():
        return
    idx = -1
    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks[0]
    if idx == -1:
        return
    # Graph Construction
    kf_idx = []
    # k to previous consecutive keyframes
    n_consec = 1
    for j in range(min(n_consec, idx)):
        kf_idx.append(idx - 1 - j)
    frame = keyframes[idx]

    # find local(!) co-visible frames
    retrieval_inds = retrieval_database.update(
        frame,
        add_after_query=True,
        k=config["retrieval"]["k"],
        min_thresh=config["retrieval"]["min_thresh"],
    )

    retrieval_inds_selected = []
    retrieval_inds = find_valid_numbers(idx,retrieval_inds)

    for kkk in retrieval_inds:
        if np.fabs(idx - kkk) < 20:
            retrieval_inds_selected.append(kkk)
    kf_idx += retrieval_inds_selected

    lc_inds = set(retrieval_inds)
    lc_inds.discard(idx - 1)
    if len(lc_inds) > 0:
        print("Database retrieval", idx, ": ", lc_inds)

    kf_idx = set(kf_idx)  # Remove duplicates by using set
    kf_idx.discard(idx)  # Remove current kf idx if included
    kf_idx = list(kf_idx)  # convert to list
    frame_idx = [idx] * len(kf_idx)
    
    print('[INFO] add factor',time.time())
    if kf_idx:
        factor_graph.add_factors(
            kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
        )
    print('[INFO] add factor.',time.time())

    with states.lock:
        states.edges_ii[:] = factor_graph.ii.cpu().tolist()
        states.edges_jj[:] = factor_graph.jj.cpu().tolist()

    factor_graph.solve_GN_calib(config["use_calib"])
    
    # the fisrt time that VI init is finished
    # transform current states
    if factor_graph.init_vi_signal:
        factor_graph.solve_GN_calib(config["use_calib"])
        factor_graph.init_vi_signal = False
        states.T_WC[:] = factor_graph.frames.last_keyframe().T_WC[:].data

        for i in range(int(keyframes.n_size.value)):
            frame_id = keyframes.dataset_idx[i].item()
            dd = keyframes.T_WC[i].data.cpu().numpy()[0]
            bb = factor_graph.bs[i].vector()
            factor_graph.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d 1\n' % (factor_graph.poses_stamps[frame_id],
                                                                                 dd[0].item(),
                                                                                 dd[1].item(),
                                                                                 dd[2].item(),
                                                                                 dd[3].item(),
                                                                                 dd[4].item(),
                                                                                 dd[5].item(),
                                                                                 dd[6].item(),
                                                                                 dd[7].item(),
                                                                                 bb[0],bb[1],bb[2],
                                                                                 bb[3],bb[4],bb[5],
                                                                                 frame_id))
            factor_graph.fp.flush()


    with states.lock:
        if len(states.global_optimizer_tasks) > 0:
            idx = states.global_optimizer_tasks.pop(0)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="config/intrinsics_zyx.yaml")
    parser.add_argument("--imu_path", default="")
    parser.add_argument("--imu_dt", type = float, default=-0.0)
    parser.add_argument("--stamp_path", default="")
    parser.add_argument("--result_path", default="result.txt")
    parser.add_argument("--start_from", type =  int, default=0)
    parser.add_argument("--end_at", type =  int, default=-1)
    parser.add_argument("--save_h5", action="store_true")


    args = parser.parse_args()
    load_config(args.config)


    if args.save_h5:
        f_h5 = h5py.File('data.h5', "w")
    
    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset,args.stamp_path)
    dataset.subsample(config["dataset"]["subsample"],args.start_from,args.end_at)
    h, w = dataset.get_img_shape()[0]
    
    if args.calib and config["use_calib"]:
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
    if not (intrinsics.get("height_new",None) is None):
        h = intrinsics.get("height_new",None) * w // intrinsics["width"]

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
        )
        viz.start()

    model = load_mast3r(device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"\

        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg()

    factor_graph = FactorGraph(model, keyframes, K, device, args)
    factor_graph.poses_stamps = dataset.timestamps
    
    retrieval_database = load_retriever(model)

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg
        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        # time.sleep(0.2)
        if save_frames:
            frames.append(img)


        TSim3 = lietorch.Sim3.Identity(1, device='cpu')
        Tic0 = np.array([1, 0,  0, 0,
                         0, 0,  1, 0,
                         0,-1,  0, 0,
                         0, 0,  0, 1]).reshape([4,4]) 
        TTTc = Tic0
        qqq = Rotation.from_matrix(TTTc[0:3,0:3]).as_quat()
        TSim3[0].data[0] = TTTc[0,3]
        TSim3[0].data[1] = TTTc[1,3]
        TSim3[0].data[2] = TTTc[2,3]
        TSim3[0].data[3] = qqq[0]
        TSim3[0].data[4] = qqq[1]
        TSim3[0].data[5] = qqq[2]
        TSim3[0].data[6] = qqq[3]
        # get frames last camera pose
        T_WC = (
            TSim3
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference, and encoded features neeed for database
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1 + keyframes.rollup_sum.value)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)
            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)
        elif mode == Mode.RELOC:
            X, C = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
        else:
            raise Exception("Invalid mode")
        
        # using IMU prediction to adjust keyframe selectiion
        if factor_graph.enable_ms and frame.frame_id>100:
            dd_old = keyframes.last_keyframe().T_WC.data.cpu().numpy()[0]
            dd_new = states.T_WC[0].data.cpu().numpy()
            dT, wTc_pred, pred_dt = factor_graph.predict_pose(frame.frame_id)
            if pred_dt > 5.0: # if prediction is too long, just use visual tracking
                pass #do nothing
            else:
                if (not add_new_kf) and  np.linalg.norm(Rotation.from_matrix(dT[0:3,0:3]).as_rotvec())>30.0/57.3:
                    add_new_kf = True
                    tracker.reset_idx_f2k()
                if add_new_kf and (np.linalg.norm(dT[0:3,3]) < 1.0 and np.linalg.norm(Rotation.from_matrix(dT[0:3,0:3]).as_rotvec())<5.0/57.3):
                    add_new_kf = False
                    tracker.idx_f2k = tracker.idx_f2k_backup
    
        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1 + keyframes.rollup_sum.value)
        print('[INFO] backend',time.time())
        run_backend(states, keyframes)
        
        print(factor_graph.frames_to_save)
        if args.save_h5:
            for iframe in factor_graph.frames_to_save:
                frame_temp = keyframes[iframe] 
                buffer = io.BytesIO()
                torch.save({
                    'feat': frame_temp.feat.cpu(), 
                    'pos': frame_temp.pos.cpu(),   
                    'X': frame_temp.X_canon.cpu(),
                    'C': frame_temp.C.cpu(),
                    'K': frame_temp.K.cpu(),
                    'N': frame_temp.N,
                    'uimg': (frame_temp.uimg * 255).to(torch.uint8).cpu().numpy(),
                    'img_shape': frame_temp.img_shape.cpu(),
                    'T_WC': frame_temp.T_WC.data.cpu(),
                    'id': frame_temp.frame_id,
                }, buffer)
                buffer.seek(0)
                f_h5.create_dataset(f"frame_{iframe}", data=np.void(buffer.read()))
        factor_graph.frames_to_save = []



        # write results
        dd = states.T_WC[0].data.cpu().numpy() # visual tracking
        frame_id = frame.frame_id
        try:
            bb = factor_graph.bs[-1].vector()
        except:
            bb = np.zeros(6)
        if factor_graph.enable_ms and frame.frame_id>100 and 'wTc_pred' in locals() and pred_dt < 5.0: # IMU prediction
            dd = np.concatenate([wTc_pred[0:3,3],Rotation.from_matrix(wTc_pred[0:3,0:3]).as_quat(),np.array([1.0])])
        factor_graph.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d 0\n' % (factor_graph.poses_stamps[frame_id],
                                                                             dd[0].item(),
                                                                             dd[1].item(),
                                                                             dd[2].item(),
                                                                             dd[3].item(),
                                                                             dd[4].item(),
                                                                             dd[5].item(),
                                                                             dd[6].item(),
                                                                             dd[7].item(),
                                                                             bb[0],bb[1],bb[2],
                                                                             bb[3],bb[4],bb[5],
                                                                             frame_id))
        factor_graph.fp.flush()
        
        if add_new_kf:
            dd = keyframes.last_keyframe().T_WC.data.cpu().numpy()[0]
            frame_id = keyframes.last_keyframe().frame_id
            bb = factor_graph.bs[-1].vector()
            factor_graph.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d 1\n' % (factor_graph.poses_stamps[frame_id],
                                                                                 dd[0].item(),
                                                                                 dd[1].item(),
                                                                                 dd[2].item(),
                                                                                 dd[3].item(),
                                                                                 dd[4].item(),
                                                                                 dd[5].item(),
                                                                                 dd[6].item(),
                                                                                 dd[7].item(),
                                                                                 bb[0],bb[1],bb[2],
                                                                                 bb[3],bb[4],bb[5],
                                                                                 frame_id))
            factor_graph.fp.flush()

        # run_backend_orig(states, keyframes)
        print('[INFO] backend.',time.time())


        # handling sliding window
        # notice that we main very few frames to save GPU memory usage
        # generally 8 GB is enough
        if len(keyframes) > 30:
            keyframes.roll_up(15)

        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1


    # finally 
    last_pin = factor_graph.get_unique_kf_idx()[-1]
    for iframe in range(factor_graph.last_pin,last_pin+1):
        frame_temp = keyframes[iframe] 
        buffer = io.BytesIO()
        torch.save({
            'feat': frame_temp.feat.cpu(), 
            'pos': frame_temp.pos.cpu(),   
            'X': frame_temp.X_canon.cpu(),
            'C': frame_temp.C.cpu(),
            'K': frame_temp.K.cpu(),
            'N': frame_temp.N,
            'uimg': (frame_temp.uimg * 255).to(torch.uint8).cpu().numpy(),
            'img_shape': frame_temp.img_shape.cpu(),
            'T_WC': frame_temp.T_WC.data.cpu(),
            'id': frame_temp.frame_id,
        }, buffer)
        buffer.seek(0)
        f_h5.create_dataset(f"frame_{iframe}", data=np.void(buffer.read()))

    factor_graph.save_graph('graph.pkl')

    # if dataset.save_results:
    #     save_dir, seq_name = eval.prepare_savedir(args, dataset)
    #     eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
    #     eval.save_reconstruction(
    #         save_dir,
    #         f"{seq_name}.ply",
    #         keyframes,
    #         last_msg.C_conf_threshold,
    #     )
    #     eval.save_keyframes(
    #         save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
    #     )
    # if save_frames:
    #     savedir = pathlib.Path(f"logs/frames/{datetime_now}")
    #     savedir.mkdir(exist_ok=True, parents=True)
    #     for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
    #         frame = (frame * 255).clip(0, 255)
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #         cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    states.set_mode(Mode.TERMINATED)
    if not args.no_viz:
        viz.join()

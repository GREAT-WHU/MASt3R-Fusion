import h5py
import io
import torch
from mast3r_fusion.config import load_config, config, set_global_config
from mast3r_fusion.mast3r_utils import mast3r_match_symmetric
from mast3r_fusion.mast3r_utils import (
    load_mast3r,
    load_retriever,
    mast3r_inference_mono,
)
from mast3r_fusion.frame import Frame
import lietorch
import tqdm
import numpy as np

import matplotlib.pyplot as plt
from mast3r_fusion.global_opt import Align2GTSAM_factors, getPosesRel, getPoses
from mast3r_fusion.geometry import (
    constrain_points_to_ray,
)
import mast3r_fusion_backends
import gtsam
import gtsam_unstable
from gtsam.symbol_shorthand import B, V, X, S, Z, C
import pickle
import argparse

import matplotlib
import os

from scipy.interpolate import interp1d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loop closure with mast3r_fusion + GTSAM")
    parser.add_argument(
        "--h5_file",
        type=str,
        default="data_90800.h5",
        help="Input H5 dataset file"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/base.yaml",
        help="config file"
    )
    parser.add_argument(
        "--loop_output",
        type=str,
        default="graph_loop_90800.pkl",
        help="Output pickle file for loop closure factors"
    )
    parser.add_argument(
        "--save_loop_viz",
        action="store_true"
    )


    args = parser.parse_args()

    if args.save_loop_viz:
        if not os.path.exists('temp'):
            os.mkdir('temp')
        print('Loop visualization is saved to ./temp')

    H5_FILE = args.h5_file
    loop_path = args.loop_output


    model = load_mast3r(device='cuda')
    model.share_memory()
    retrieval_database = load_retriever(model)
    load_config(args.config)
    all_factors = []


    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))
        Cs = torch.stack([kf.get_average_conf() for kf in kfs])
        return Xs, T_WCs, Cs

    def mini_solve(cfg, K, img_shape, Xs, T_WCs, Cs, ii,jj,idx_ii2jj,valid_match,Q_ii2jj, ii_orig, jj_orig):
        global all_factors
        C_thresh       = cfg["C_conf"]
        Q_thresh       = cfg["Q_conf"]
        pixel_border   = cfg["pixel_border"]
        z_eps          = cfg["depth_eps"]
        max_iter       = cfg["max_iters"]
        sigma_pixel    = cfg["sigma_pixel"]
        sigma_depth    = cfg["sigma_depth"]
        delta_thresh   = cfg["delta_norm"]
        img_size       = img_shape[0]
        height, width  = img_size
        K              = K
        unique_kf_idx  = torch.unique(torch.cat([ii, jj]), sorted=True)
        # Xs, T_WCs, Cs  = self.get_poses_points(unique_kf_idx)
        Xs = constrain_points_to_ray(img_size, Xs, K)
        pose_data = T_WCs.data[:, 0, :].clone()
        pose_data_temp = pose_data.clone()
        pose_data_temp[:,7] = 1.0
        wTcs_temp = lietorch.Sim3(pose_data_temp).matrix().cpu().numpy()
        R21 = wTcs_temp[1,0:3,0:3].T @ wTcs_temp[0, 0:3,0:3]
        t21 =  (np.eye(3,3) - R21) @ np.array([0,0,10.0])
        T21 = np.eye(4,4)
        T21[0:3,0:3] = R21
        T21[0:3,3] = t21
        wTcs_temp[1] =  wTcs_temp[0] @ np.linalg.inv(T21)
        # YOU NEED GOOD INITIALS FOR THIS TO WORK
        ss_temp = pose_data[:,7].cpu().numpy()


        for i in range(10):
            pose_data_new = getPosesRel(unique_kf_idx,pose_data,wTcs_temp,ss_temp,True)
            aligncore = mast3r_fusion_backends.AlignCoreCalib()
            aligncore.init(
                pose_data_new,
                Xs,
                Cs,
                K.to('cuda'),
                ii, # edge
                jj, # edge
                idx_ii2jj, # matching
                valid_match, # mask
                Q_ii2jj, # uncertainty
                height.item(),
                width.item(),
                pixel_border,
                z_eps,
                sigma_pixel,
                sigma_depth,
                C_thresh,
                Q_thresh,
                max_iter,
                delta_thresh,1,1.3
            )
            H11 = torch.zeros([4,ii.shape[0],7,7],dtype=torch.float64,device='cpu')
            v11 = torch.zeros([2,ii.shape[0],7],dtype=torch.float64,device='cpu')
            c11 = torch.zeros([ii.shape[0]],dtype=torch.float64,device='cpu')
            aligncore.hessian_pieces(H11,v11,c11)
            vfactors = Align2GTSAM_factors(H11.numpy(),v11.numpy(),wTcs_temp,ss_temp,ii.cpu().numpy(),jj.cpu().numpy(),0)
            prior_factors = []
            initials = gtsam.Values()
            cur_graph = gtsam.NonlinearFactorGraph()
            symbols = []

            for iii in unique_kf_idx:
                initials.insert(X(iii),gtsam.Pose3(wTcs_temp[iii]))
                initials.insert(S(iii),ss_temp[iii])
                symbols.append(S(iii))
                symbols.append(X(iii))
                prior_factors.append(gtsam.PriorFactorDouble(S(iii),ss_temp[iii], gtsam.noiseModel.Diagonal.Sigmas([0.0001])))

            # Visual constraint
            for h_factor in vfactors:
                cur_graph.add(h_factor)
            for factor in prior_factors:
                cur_graph.add(factor)

            params = gtsam.LevenbergMarquardtParams();params.setMaxIterations(2)
            params.setVerbosityLM("SUMMARY")
            optimizer = gtsam.LevenbergMarquardtOptimizer(cur_graph, initials, params)
            cur_result = optimizer.optimize()
            assert(T_WCs.shape[0] == pose_data.shape[0])

            for iii in unique_kf_idx:
                ss_temp[iii] = cur_result.atDouble(S(iii))
                wTcs_temp[iii] = cur_result.atPose3(X(iii)).matrix()
            pose_data_new = getPoses(unique_kf_idx,pose_data,wTcs_temp,ss_temp)
            pose_data[:,:] = pose_data_new[:,:]
        for iii in range(len(ii_orig)):
            all_factors.append({'type':'visual_loop','H':H11[0,iii],'v':v11[0,iii],'iijj':[ii_orig[iii],jj_orig[iii]],
                                                     'params':[wTcs_temp[ii[iii]],ss_temp[ii[iii]],wTcs_temp[jj[iii]],ss_temp[jj[iii]]]})

    def find_valid_numbers(a, b):
        result = []
        for i, c in enumerate(b):
            if abs(c - a) <= 5:
                continue 

            close_indices = [j for j, d in enumerate(b) if abs(d - c) <= 20]

            if i == min(close_indices) or c == a - 2 :
                result.append(c)
        return result

    def load_frame_from_h5(h5_filename, iframe):
        with h5py.File(h5_filename, "r") as f:
            blob = bytes(f[f"frame_{iframe}"][()])
            buffer = io.BytesIO(blob)
            return torch.load(buffer, map_location="cpu",weights_only=False)

    def len_h5(h5_filename):
        with h5py.File(h5_filename, "r") as f:
            return len(f.keys())

    T_WC_map = {}
    retrieval_map = {}


    x_d = []
    for i in tqdm.tqdm(range(len_h5(H5_FILE))):
        data = load_frame_from_h5(H5_FILE, i)
        T_WC_map[i] = data['T_WC']
        x_d.append(T_WC_map[i].cpu().numpy()[0,0:3])
    x_d = np.array(x_d)

    plt.figure('loop')
    x_series = []
    y_series = []
    for idx in sorted(T_WC_map.keys()):
        x_series.append(T_WC_map[idx][0][0])
        y_series.append(T_WC_map[idx][0][1])


    def gen_conf_map_vec(x_d, dl, dn):
        N = x_d.shape[0]
        values = np.zeros((N, N))
        Rr = np.array([[0, 1], [-1, 0]])

        dxs = x_d[1:, :2] - x_d[:-1, :2]       
        lsq = np.sum(dxs**2, axis=1)           
        ls = np.sqrt(lsq)
        ns = dxs / ls[:, None]                 

        n_outer = np.einsum("ni,nj->nij", ns, ns)          
        rns = ns @ Rr.T                                    
        r_outer = np.einsum("ni,nj->nij", rns, rns)        

        inc = n_outer * (lsq * (dl**2))[:, None, None] \
            + r_outer * (lsq * (dn**2))[:, None, None]    

        prefix = np.zeros((N, 2, 2))
        np.cumsum(inc, axis=0, out=prefix[1:])

        for ii in range(N):
            # P(ii→i) = prefix[i] - prefix[ii]
            P_blocks = prefix[ii+1:] - prefix[ii]       
            X = x_d[ii+1:, :2] - x_d[ii, :2]            

            # xPx / (x·x)
            xPx = (
                P_blocks[:, 0, 0] * X[:, 0]**2
                + 2 * P_blocks[:, 0, 1] * X[:, 0] * X[:, 1]
                + P_blocks[:, 1, 1] * X[:, 1]**2
            )
            denom = np.sum(X**2, axis=1)

            vals = np.sqrt(xPx / denom)
            values[ii, ii+1:] = vals
            values[ii+1:, ii] = vals 

        return values

    # plt.plot(x_d[:,0],x_d[:,1])
    # plt.show()
    def gen_conf_map(x_d):
        values = np.zeros([x_d.shape[0],x_d.shape[0]])
        Rr = np.array([[0,1],
                      [-1,0]])
        dl = 0.25
        dn = 0.10
        for ii in range(x_d.shape[0]):
            P = np.zeros([2,2])
            for i in range(ii+1,x_d.shape[0]):
                dx = x_d[i,0:2] - x_d[i-1,0:2]
                # p1 = p0 + n * l * (1 + dl) +  Rr @ n * l * dn
                # J  =      [n * l  ,   R @ n]
                n = dx/np.linalg.norm(dx)
                l = np.linalg.norm(dx)
                P += n[None].T @ n[None] * (l**2) * (dl**2) \
                     + (Rr @ n)[None].T @ (Rr @ n)[None]  * (l**2) * (dn**2)
                x = x_d[i,0:2]-x_d[ii,0:2]
                values[ii,i] = np.sqrt((P[0,0]*x[0]**2+P[1,1]*x[1]**2 + 2*x[0]*x[1]*P[0,1])\
                                                   /(x[0]**2+x[1]**2))
                values[i,ii] = values[ii,i]
        return values

    plt.figure('conf_map',figsize=[3,2.5])
    conf_map = gen_conf_map_vec(x_d, config['loop']['conf_noise_along'],config['loop']['conf_noise_cross'])
    plt.imshow(conf_map,cmap='summer')
    plt.colorbar()

    # plt.show()

    for i in tqdm.tqdm(range(len_h5(H5_FILE))):
        data = load_frame_from_h5(H5_FILE, i)
        frame = Frame(
            int(data['id']),
            None,
            data['img_shape'],
            None,
            None,
            lietorch.Sim3(data['T_WC'].to('cuda')),
        )
        frame.X_canon = data['X'].to('cuda')
        frame.C = data['C'].to('cuda')
        frame.N = data['N']
        frame.feat = data['feat'].to('cuda')
        frame.pos = data['pos'].to('cuda')
        retrieval_inds = retrieval_database.update(
        frame,
        add_after_query=True,
        k=10,
        min_thresh=0.0,
        )
        retrieval_inds_selected = []
        retrieval_inds = find_valid_numbers(i,retrieval_inds)

        for kkk in retrieval_inds:
            # candidate filtering based on ``conf_map''
            T0 = lietorch.Sim3(T_WC_map[i][0]).matrix()
            T1 = lietorch.Sim3(T_WC_map[kkk][0]).matrix()
            interest_distance = config['loop']['interest_distance']
            interest_point0 = T0[0:3,0:3] @ np.array([0,0,interest_distance]) + T0[0:3,3]
            interest_point1 = T1[0:3,0:3] @ np.array([0,0,interest_distance]) + T1[0:3,3]
            if np.fabs(i - kkk)> 20 and np.linalg.norm(interest_point0[0:2] - interest_point1[0:2]) <interest_distance + conf_map[i,kkk]:
                retrieval_inds_selected.append(kkk)
            if np.fabs(i - kkk) < 5:
                retrieval_inds_selected.append(kkk)
        retrieval_map[i] = retrieval_inds_selected

        for kkk in retrieval_inds_selected:
            data_kkk = load_frame_from_h5(H5_FILE, kkk)
            frame_kkk = Frame(
                int(data_kkk['id']),
                None,
                data_kkk['img_shape'],
                None,
                None,
                lietorch.Sim3(data_kkk['T_WC'].to('cuda')),
            )
            frame_kkk.X_canon = data_kkk['X'].to('cuda')
            frame_kkk.C = data_kkk['C'].to('cuda')
            frame_kkk.N = data_kkk['N']
            frame_kkk.feat = data_kkk['feat'].to('cuda')
            frame_kkk.pos = data_kkk['pos'].to('cuda')

            ii = torch.tensor([0])
            jj = torch.tensor([1])
            (
                idx_i2j,
                idx_j2i,
                valid_match_j,
                valid_match_i,
                Qii,
                Qjj,
                Qji,
                Qij,
            ) = mast3r_match_symmetric(
                model, data['feat'].to('cuda'), data['pos'].to('cuda'),
                       data_kkk['feat'].to('cuda'), data_kkk['pos'].to('cuda'), 
                       data['img_shape'][None], data['img_shape'][None], 1
            )



            batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
                :, None
            ].repeat(1, idx_i2j.shape[1])

            Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji)
            Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij)

            valid_Qj = Qj > config['local_opt']["Q_conf"]
            valid_Qi = Qi > config['local_opt']["Q_conf"]
            valid_j = valid_match_j & valid_Qj
            valid_i = valid_match_i & valid_Qi
            nj = valid_j.shape[1] * valid_j.shape[2]
            ni = valid_i.shape[1] * valid_i.shape[2]
            match_frac_j = valid_j.sum(dim=(1, 2)) / nj
            match_frac_i = valid_i.sum(dim=(1, 2)) / ni

            ii_tensor = torch.as_tensor(ii, device='cuda')
            jj_tensor = torch.as_tensor(jj, device='cuda')

            # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
            invalid_edges = torch.minimum(match_frac_j, match_frac_i) < 0.03

            valid_edges = ~invalid_edges
            if torch.sum(valid_edges) == 0:continue
            ii_tensor = ii_tensor[valid_edges]
            jj_tensor = jj_tensor[valid_edges]
            idx_i2j = idx_i2j[valid_edges]
            idx_j2i = idx_j2i[valid_edges]
            valid_match_j = valid_match_j[valid_edges]
            valid_match_i = valid_match_i[valid_edges]
            Qj[invalid_edges,:] *= 0.0001
            Qi[invalid_edges,:] *= 0.0001
            Qj = Qj[valid_edges]
            Qi = Qi[valid_edges]


            ii_two = torch.cat((ii_tensor, jj_tensor), dim=0)
            jj_two = torch.cat((jj_tensor, ii_tensor), dim=0)
            idx_ii2jj = torch.cat((idx_i2j,idx_j2i), dim=0)
            valid_match = torch.cat((valid_match_j, valid_match_i), dim=0)
            Q_ii2jj = torch.cat((Qj, Qi), dim=0)

            if args.save_loop_viz:
                iiii = 0
                if i < len(x_series) - 1: 
                    plt.figure('1',figsize=[10,3])
                    plt.subplot(1,3,1)
                    plt.plot(x_series,y_series,c=[0.8,0.8,0.8])
                    A = (x_series[i],y_series[i])
                    B = (x_series[i+1],y_series[i+1])
                    L = np.linalg.norm(np.array(A) - np.array(B))
                    plt.arrow(A[0], A[1],
                        (B[0]-A[0])/L*3, (B[1]-A[1])/L*3,
                        head_width=1, head_length=1, fc='red', ec='red',linewidth=2,zorder=1000)
                    A = (x_series[kkk],y_series[kkk])
                    B = (x_series[kkk+1],y_series[kkk+1])
                    L = np.linalg.norm(np.array(A) - np.array(B))
                    plt.arrow(A[0], A[1],
                        (B[0]-A[0])/L*3, (B[1]-A[1])/L*3,
                        head_width=1, head_length=1, fc='green', ec='green',linewidth=2,zorder=1000)
                    plt.xlim([A[0]-30,A[0]+30])
                    plt.ylim([A[1]-30,A[1]+30])

                    plt.subplot(1,3,2)
                    plt.imshow(data['uimg'])

                    mask = valid_match_j[iiii,::100,0].cpu().numpy()
                    pts = idx_ii2jj[iiii,::100].cpu().numpy()
                    clr = np.arange(valid_match_j.shape[1])[::100]
                    pts = pts[mask]
                    clr = clr[mask]

                    plt.scatter(pts % 512,pts // 512,s=1,c=clr,cmap='jet')

                    plt.subplot(1,3,3)
                    plt.imshow(data_kkk['uimg'])
                    pts1 = np.arange(valid_match_j.shape[1])[::100]
                    pts1 =pts1[mask]
                    plt.scatter(pts1 % 512,pts1 // 512,s=1,c=clr,cmap='jet')
                    plt.savefig('temp/loop_%d_%d.jpg'%(i,kkk),dpi=300)
                    plt.close('all')

                iiii = 1
                if i < len(x_series) - 1: 
                    plt.figure('1',figsize=[10,3])
                    plt.subplot(1,3,1)
                    plt.plot(x_series,y_series,c=[0.8,0.8,0.8])
                    A = (x_series[i],y_series[i])
                    B = (x_series[i+1],y_series[i+1])
                    L = np.linalg.norm(np.array(A) - np.array(B))
                    plt.arrow(A[0], A[1],
                        (B[0]-A[0])/L*3, (B[1]-A[1])/L*3,
                        head_width=1, head_length=1, fc='red', ec='red',linewidth=2,zorder=1000)
                    A = (x_series[kkk],y_series[kkk])
                    B = (x_series[kkk+1],y_series[kkk+1])
                    L = np.linalg.norm(np.array(A) - np.array(B))
                    plt.arrow(A[0], A[1],
                        (B[0]-A[0])/L*3, (B[1]-A[1])/L*3,
                        head_width=1, head_length=1, fc='green', ec='green',linewidth=2,zorder=1000)
                    plt.xlim([A[0]-30,A[0]+30])
                    plt.ylim([A[1]-30,A[1]+30])

                    plt.subplot(1,3,2)
                    plt.imshow(data['uimg'])
                    mask = valid_match[iiii,::100,0].cpu().numpy()

                    pts = idx_ii2jj[iiii,::100].cpu().numpy()
                    clr = np.arange(valid_match.shape[1])[::100]
                    pts = pts[mask]
                    clr = clr[mask]

                    pts1 = np.arange(valid_match.shape[1])[::100]
                    pts1 =pts1[mask]
                    plt.scatter(pts1 % 512,pts1 // 512,s=1,c=clr,cmap='jet')

                    plt.subplot(1,3,3)
                    plt.imshow(data_kkk['uimg'])

                    plt.scatter(pts % 512,pts // 512,s=1,c=clr,cmap='jet')

                    plt.savefig('temp/loop_%d_%d_.jpg'%(i,kkk),dpi=300)
                    plt.close('all')


            Xs = torch.stack([kf.X_canon for kf in [frame,frame_kkk]])
            T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in [frame,frame_kkk]]))
            Cs = torch.stack([kf.get_average_conf() for kf in [frame,frame_kkk]])
            mini_solve(config['local_opt'],data['K'],data['img_shape'],Xs,T_WCs,Cs,ii_two,jj_two,idx_ii2jj,valid_match,Q_ii2jj, [i,kkk],[kkk,i])

    pickle.dump(all_factors,open(loop_path,'wb'))

    plt.figure('loop',figsize=[4,4])
    for idx in sorted(T_WC_map.keys()):
        for kkk in retrieval_map[idx]:
            plt.plot([T_WC_map[idx][0][0],T_WC_map[kkk][0][0]],
                     [T_WC_map[idx][0][1],T_WC_map[kkk][0][1]],c='red',linewidth=1,zorder=10000)
    plt.plot(x_series,y_series,c='black',linewidth=1)
    plt.savefig('loop.svg')

    print('Close the visualization windows to continue...')
    plt.show()

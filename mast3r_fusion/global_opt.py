import lietorch
import torch
from mast3r_fusion.config import config
from mast3r_fusion.frame import SharedKeyframes
from mast3r_fusion.geometry import (
    constrain_points_to_ray,
)
from mast3r_fusion.mast3r_utils import mast3r_match_symmetric
import mast3r_fusion_backends

import numpy as np
import gtsam
import gtsam_unstable
from gtsam.symbol_shorthand import B, V, X, S, Z, C
from scipy.spatial.transform import Rotation
import mast3r_fusion.geoFunc.trans as trans
import mast3r_fusion.geoFunc.data_utils as data_utils
import os
import math
import time
import yaml
import pickle
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from .vio_utils import VisualIMUAlignment, keys2str, skew_sym

BA2GTSAM_PYTHON_IMPLEMENTATION = False

if not BA2GTSAM_PYTHON_IMPLEMENTATION:
    # C++ implementation
    def Align2GTSAM_factors(H11: np.ndarray, v11: np.ndarray, wTcs, ss, ii, jj, pin):
        factors = gtsam.Align2GTSAM_factors(list(H11.reshape(-1, 7,7)),
                                         list(v11.reshape(-1, 7)),wTcs,ss,ii,jj,pin)
        return factors
else:
    def Align2GTSAM_factors(H11: np.ndarray, v11: np.ndarray, wTcs, ss, ii, jj, pin):
        factors = []
        for idx in range(ii.shape[0]):
            i = ii[idx] - pin
            j = jj[idx] - pin

            Xi = np.copy(wTcs[i])
            Xi[0:3,0:3] *= ss[i]
            Xj = np.copy(wTcs[j])
            Xj[0:3,0:3] *= ss[j]
            Xij = np.linalg.inv(Xi) @ Xj

            s = np.power(np.linalg.det(Xij[0:3,0:3]),1.0/3)
            R = Xij[0:3,0:3]/s
            t = Xij[0:3,3]
            pXij_pXj = np.zeros([7,7])
            pXij_pXj[0:3,0:3] = s * R
            pXij_pXj[0:3,3:6] = skew_sym(t) @ R
            pXij_pXj[0:3,6] = -t
            pXij_pXj[3:6,3:6] = R
            pXij_pXj[6,6] = 1

            s = ss[j]
            pXj_pXj = np.zeros([7,7])
            pXj_pXj[0,6] = s
            pXj_pXj[4:7,0:3] = s*np.eye(3,3)
            pXj_pXj[1:4,3:6] = np.eye(3,3)
            pXij_pXj_ = pXij_pXj@np.linalg.inv(pXj_pXj)

            pXij_pXi = -np.eye(7,7)
            s = ss[i]
            pXi_pXi = np.zeros([7,7])
            pXi_pXi[0,6] = s
            pXi_pXi[4:7,0:3] = s*np.eye(3,3)
            pXi_pXi[1:4,3:6] = np.eye(3,3)
            pXij_pXi_ = pXij_pXi@np.linalg.inv(pXi_pXi)

            J = np.hstack([pXij_pXi_,pXij_pXj_])
            H = H11[0,idx,:,:]
            v = v11[0,idx,:]
            HHH = J.T @ H @ J
            vvv = J.T @ v

            symbols = [S(i),X(i),S(j),X(j)]
            initials = gtsam.Values()
            initials.insert(S(i),ss[i])
            initials.insert(S(j),ss[j])
            initials.insert(X(i),gtsam.Pose3(wTcs[i]))
            initials.insert(X(j),gtsam.Pose3(wTcs[j]))
            factors.append(CustomHessianFactor(symbols,initials,HHH/1e6,-vvv/1e6))
        return factors

    def CustomHessianFactor(symbols, values, H: np.ndarray, v: np.ndarray):
        info_expand = np.zeros([H.shape[0]+1,H.shape[1]+1])
        info_expand[0:-1,0:-1] = H
        info_expand[0:-1,-1] = v
        info_expand[-1,-1] = 100.0 # This is meaningless.
        dims = []
        for sym in symbols:
            if sym - X(0) < 100000 and sym >= X(0):
                dims.append(6)
            if sym - S(0) < 100000 and sym >= S(0):
                dims.append(1)
        h_f = gtsam.HessianFactor(symbols,dims,info_expand)
        l_c = gtsam.LinearContainerFactor(h_f,values)
        return l_c

def getPoses(indice,pose_data,wTcs,ss):
    all_cs = []
    for iii in range(0,pose_data.shape[0]):
        T_temp = wTcs[indice[iii]]
        dd = np.concatenate([T_temp[0:3,3],Rotation.from_matrix(T_temp[0:3,0:3]).as_quat(),np.array([ss[indice[iii]]])])
        all_cs.append(torch.tensor(dd.astype(np.float32),device='cuda'))
    return torch.stack(all_cs)

def getPosesRel(indice,pose_data,wTcs,ss,enable_ms):
    all_cs = []
    for iii in range(0,pose_data.shape[0]):
        T_temp = wTcs[indice[iii]]
        dd = np.concatenate([T_temp[0:3,3],Rotation.from_matrix(T_temp[0:3,0:3]).as_quat(),np.array([ss[indice[iii]]])])
        all_cs.append(torch.tensor(dd.astype(np.float64),device='cuda'))
    lll = lietorch.Sim3(torch.stack(all_cs))
    if enable_ms:
        llll = all_cs[0].clone()
        T_0 = lietorch.Sim3(llll)
        for iii in range(0,pose_data.shape[0]):
            LLL = T_0.inv() * lll[iii]
            lll.data[iii,:] = LLL.data[:]
    return lll.data.to(device = 'cuda',dtype=torch.float32)

class FactorGraph:
    def __init__(self, model, frames: SharedKeyframes, K=None, device="cuda", args = None):
        self.model = model
        self.frames = frames
        self.device = device
        self.cfg = config["local_opt"]
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)

        self.K = K

        self.init_vi_signal = False
        self.enable_ms = False
        self.viz_matching = False


        self.enable_excalib = config["ms_opt"]['enable_excalib']
        self.subpixel_factor = config["ms_opt"]['subpixel_factor']
        self.d_diff_threshold = config["ms_opt"]['d_diff_threshold']
        self.window_num = config["ms_opt"]['window_num']
        self.retain_num = self.window_num + 10 # reserved for marginalization
        self.frames_to_save = []

        calib = yaml.load(open(args.calib,'rt'), Loader=yaml.SafeLoader)
        self.poses_ref = {}
        self.poses_stamps = {}
        self.Tic = np.copy(calib['Tic'])
        self.wTcs           = []
        self.ss             = []
        self.vs             = []
        self.bs             = []
        self.preintegrations= []
        self.last_pin = 0
        self.marg_factor = None
        self.cur_graph   = None
        self.cur_result  = None
        # sliding window related

        self.init_bias_noise = np.array(config['ms_opt']['init_bias_noise'])
        self.regularization_noise = np.array(config['ms_opt']['regularization_noise'])
        noise = np.array(config['ms_opt']['imu_noise'])
        accel_noise_sigma = noise[0]
        gyro_noise_sigma = noise[1]
        accel_bias_rw_sigma = noise[2]
        gyro_bias_rw_sigma = noise[3]
        GRAVITY = 9.81
        measured_acc_cov = np.eye(3,3) * math.pow(accel_noise_sigma,2)
        measured_omega_cov = np.eye(3,3) * math.pow(gyro_noise_sigma,2)
        integration_error_cov = np.eye(3,3) * 0e-8
        bias_acc_cov = np.eye(3,3) * math.pow(accel_bias_rw_sigma,2)
        bias_omega_cov = np.eye(3,3) * math.pow(gyro_bias_rw_sigma,2)
        bias_acc_omega_init = np.eye(6,6) * 0e-5

        params = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
        params.setAccelerometerCovariance(measured_acc_cov)
        params.setIntegrationCovariance(integration_error_cov)
        params.setGyroscopeCovariance(measured_omega_cov)
        params.setBiasAccCovariance(bias_acc_cov)
        params.setBiasOmegaCovariance(bias_omega_cov)
        params.setBiasAccOmegaInit(bias_acc_omega_init)
        self.params = params

        measured_acc_cov = np.eye(3,3) * math.pow(accel_noise_sigma,2) * 100000
        measured_omega_cov = np.eye(3,3) * math.pow(gyro_noise_sigma,2) * 100000
        params_loose = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
        params_loose.setAccelerometerCovariance(measured_acc_cov)
        params_loose.setIntegrationCovariance(integration_error_cov)
        params_loose.setGyroscopeCovariance(measured_omega_cov)
        params_loose.setBiasAccCovariance(bias_acc_cov)
        params_loose.setBiasOmegaCovariance(bias_omega_cov)
        params_loose.setBiasAccOmegaInit(bias_acc_omega_init)
        self.params_loose = params_loose


        if config['ms_opt']['imu_format'] == 'custom_deg':
            try:
                self.imu_pool = data_utils.IMUPool(np.loadtxt(args.imu_path,delimiter=' '), degree = True, dt = args.imu_dt)
            except:
                self.imu_pool = data_utils.IMUPool(np.loadtxt(args.imu_path,delimiter=','), degree = True, dt = args.imu_dt)
        elif config['ms_opt']['imu_format'] == 'custom_rad':
            try:
                self.imu_pool = data_utils.IMUPool(np.loadtxt(args.imu_path,delimiter=' '), degree = False, dt = args.imu_dt)
            except:
                self.imu_pool = data_utils.IMUPool(np.loadtxt(args.imu_path,delimiter=','), degree = False, dt = args.imu_dt)
        elif config['ms_opt']['imu_format'] == 'subt':
            all_imu = np.loadtxt(args.imu_path,delimiter=',',comments='#',skiprows=1)
            all_imu[:,0] /= 1e9
            all_imu_new = np.zeros_like(all_imu)
            all_imu_new[:,0] = all_imu[:,0]
            all_imu_new[:,1:4] = all_imu[:,5:8] * 180/math.pi
            all_imu_new[:,4:7] = all_imu[:,8:11]
            all_imu_new = all_imu_new[:,:7]
            self.imu_pool = data_utils.IMUPool(all_imu_new, degree = True, dt = args.imu_dt)

        self.fp = open(args.result_path,'wt')
        self.fp_id = 0
        self.transform_world = False

        self.all_factors = []


    def save_graph(self, path):

        fix_noise = 1e-6

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]
        img_size = self.frames.last_keyframe().img.shape[-2:]
        height, width = img_size

        K = self.K
        pin = self.cfg["pin"]
        pin = 0
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return

        pin = unique_kf_idx[-1].item()
        print('[INFO] marg',time.time())
        print('len factors:', len(self.all_factors))
        if pin > self.last_pin:
            print('Marginalization!!!',pin,self.last_pin)
            # Marginalization
            marg_graph = gtsam.NonlinearFactorGraph()
            Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx[self.last_pin:])
            img_size = self.frames.last_keyframe().img.shape[-2:]

            Xs = constrain_points_to_ray(img_size, Xs, K)
            
            ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()


        
            marg_mask = torch.logical_and(torch.logical_and(ii >= self.last_pin,jj>=self.last_pin),torch.logical_or(ii < pin,jj<pin))
            ii = ii[marg_mask]
            jj = jj[marg_mask]
            idx_ii2jj = idx_ii2jj[marg_mask]
            valid_match = valid_match[marg_mask]
            Q_ii2jj = Q_ii2jj[marg_mask]
            newest =torch.max(torch.max(ii),torch.max(jj))

            new_pin = pin
            pin = self.last_pin

            if not(self.marg_factor is None):
                ssss = keys2str(self.marg_factor.keys())
                for sss in ssss:
                    if 'X' in sss and int(sss[1:])+pin > newest:
                        newest = int(sss[1:])+pin
            
            T_WCs = T_WCs[:newest-pin+1]
            Xs = Xs[:newest-pin+1]
            Cs = Cs[:newest-pin+1]

            pose_data = T_WCs.data[:, 0, :]
            pose_data_new = getPosesRel(np.arange(pin,pin+pose_data.shape[0]),pose_data,self.wTcs,self.ss,self.enable_ms)
            aligncore = mast3r_fusion_backends.AlignCoreCalib()
            aligncore.init(
                pose_data_new,
                Xs,
                Cs,
                K,
                ii, # edge
                jj, # edge
                idx_ii2jj, # matching
                valid_match, # mask
                Q_ii2jj, # uncertainty
                height,
                width,
                pixel_border,
                z_eps,
                sigma_pixel,
                sigma_depth,
                C_thresh,
                Q_thresh,
                max_iter,
                delta_thresh,
                self.subpixel_factor, self.d_diff_threshold
            )

            H = torch.zeros([(pose_data.shape[0])*7,(pose_data.shape[0])*7],dtype=torch.float64,device='cpu')
            v = torch.zeros([(pose_data.shape[0])*7],dtype=torch.float64,device='cpu')

            H11 = torch.zeros([1,ii.shape[0],7,7],dtype=torch.float64,device='cpu')
            v11 = torch.zeros([1,ii.shape[0],7],dtype=torch.float64,device='cpu')
            c11 = torch.zeros([ii.shape[0]],dtype=torch.float64,device='cpu')
            aligncore.hessian_pieces(H11,v11,c11)
            vfactors = Align2GTSAM_factors(H11.numpy(),v11.numpy(),self.wTcs[pin:],self.ss[pin:],ii.cpu().numpy(),jj.cpu().numpy(),pin)
            for iii in range(H11.shape[1]):
                self.all_factors.append({'type':'visual','H':H11[0,iii],'v':v11[0,iii],'iijj':[ii[iii].item(),jj[iii].item()],
                                         'tstamps':[self.poses_stamps[self.frames[ii[iii]].frame_id],self.poses_stamps[self.frames[jj[iii]].frame_id]],
                                         'params':[self.wTcs[ii[iii]],self.ss[ii[iii]],self.wTcs[jj[iii]],self.ss[jj[iii]]]})
                self.all_factors.append({'type':'param','ii':ii[iii].item(),'v':self.vs[ii[iii]],'b':self.bs[ii[iii]]})
        pickle.dump(self.all_factors,open(path,'wb'))

    def add_factors(self, ii, jj, min_match_frac, is_reloc=False):
        # print('1',time.time())
        kf_ii = [self.frames[idx] for idx in ii]
        kf_jj = [self.frames[idx] for idx in jj]
        feat_i = torch.cat([kf_i.feat for kf_i in kf_ii])
        feat_j = torch.cat([kf_j.feat for kf_j in kf_jj])
        pos_i = torch.cat([kf_i.pos for kf_i in kf_ii])
        pos_j = torch.cat([kf_j.pos for kf_j in kf_jj])
        shape_i = [kf_i.img_true_shape for kf_i in kf_ii]
        shape_j = [kf_j.img_true_shape for kf_j in kf_jj]
        # print('2',time.time())

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
            self.model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j, self.subpixel_factor
        )
        # print('3',time.time())

        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])

        w = shape_i[0][0,1].item()
        idx_i2j_orig = idx_i2j.clone()
        idx_i2j_orig = (idx_i2j_orig // (self.subpixel_factor * w))//self.subpixel_factor * w  + (idx_i2j_orig % (self.subpixel_factor * w))//self.subpixel_factor
        idx_j2i_orig = idx_j2i.clone()
        idx_j2i_orig = (idx_j2i_orig // (self.subpixel_factor * w))//self.subpixel_factor * w  + (idx_j2i_orig % (self.subpixel_factor * w))//self.subpixel_factor
        
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j_orig] * Qji)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i_orig] * Qij)

        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj
        valid_i = valid_match_i & valid_Qi
        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni
        # print('4',time.time())

        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)

        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        invalid_edges_orig = invalid_edges.clone()
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges
        # print('5',time.time())

        if invalid_edges.any() and is_reloc:
            return False

        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        # valid_match_j = valid_j[valid_edges]
        # valid_match_i = valid_i[valid_edges]
        valid_match_j = valid_match_j[valid_edges]
        valid_match_i = valid_match_i[valid_edges]
        Qj[invalid_edges_orig,:] *= 0.0001
        Qi[invalid_edges_orig,:] *= 0.0001
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]
        # print('6',time.time())

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])
        

        # Saving matching resutls
        if  self.viz_matching:
            if not os.path.exists('temp'):
                os.mkdir('temp')
            for iiii in range(ii_tensor.shape[0]):
                plt.figure('1',figsize=[5,6])
                plt.subplot(2,1,1)
                plt.imshow(self.frames[self.ii[iiii].item()].uimg)

                mask = self.valid_match_j[iiii,::100,0].cpu().numpy()
                pts = self.idx_ii2jj[iiii,::100].cpu().numpy()
                clr = np.arange(self.valid_match_j.shape[1])[::100]
                pts = pts[mask]
                clr = clr[mask]

                plt.scatter(pts % 512,pts // 512,s=0.7,c=clr,cmap='jet')
                plt.gca().tick_params(labelbottom=False, labelleft=False)

                plt.subplot(2,1,2)
                plt.imshow(self.frames[self.jj[iiii].item()].uimg)
                pts1 = np.arange(self.valid_match_j.shape[1])[::100]
                pts1 =pts1[mask]
                plt.scatter(pts1 % 512,pts1 // 512,s=0.7,c=clr,cmap='jet')
                plt.gca().tick_params(labelbottom=False, labelleft=False)
                plt.tight_layout()
                plt.savefig('temp/%d_%d.jpg'%(self.ii[iiii].item(),self.jj[iiii].item()))
                plt.close('all')

        retain_mask = torch.logical_not(torch.logical_and(self.ii<torch.max(self.ii)-20,self.jj<torch.max(self.jj)-self.retain_num))
        self.ii = self.ii[retain_mask]
        self.jj = self.jj[retain_mask]
        self.idx_ii2jj = self.idx_ii2jj[retain_mask]
        self.idx_jj2ii = self.idx_jj2ii[retain_mask]
        self.valid_match_j = self.valid_match_j[retain_mask]
        self.valid_match_i = self.valid_match_i[retain_mask]
        self.Q_ii2jj = self.Q_ii2jj[retain_mask]
        self.Q_jj2ii = self.Q_jj2ii[retain_mask]

        added_new_edges = valid_edges.sum() > 0
        return added_new_edges

    def get_unique_kf_idx(self):
        # return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)
        if len(self.ii) == 0:
            return torch.tensor([],dtype=self.ii.dtype,device=self.ii.device)
        return torch.arange(0,torch.max(torch.cat([self.ii, self.jj])+1))

    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx] for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        T_WCs = lietorch.Sim3(torch.stack([kf.T_WC.data for kf in kfs]))

        Cs = torch.stack([kf.get_average_conf() for kf in kfs])

        return Xs, T_WCs, Cs


    def predict_pose(self,frame_id,kf_idx=-1):
        if kf_idx == -1:
            kf_idx = len(self.bs)-1
        new_preintegration =  gtsam.PreintegratedCombinedMeasurements(self.params,self.bs[kf_idx])
        dd = self.imu_pool.get_records(self.poses_stamps[self.frames[kf_idx].frame_id],
                                       self.poses_stamps[frame_id])
        for t0, t1, ddd in dd:
            new_preintegration.integrateMeasurement(ddd[3:6],ddd[0:3]/180*math.pi,t1-t0)
        print(self.vs[kf_idx])
        wTi_pred = new_preintegration.predict(gtsam.NavState(gtsam.Pose3(self.wTcs[kf_idx] @ np.linalg.inv(self.Tic)), self.vs[kf_idx]), self.bs[kf_idx]).pose().matrix()
        dT = np.linalg.inv(self.wTcs[kf_idx] @ np.linalg.inv(self.Tic)) @ wTi_pred
        return dT, wTi_pred @ self.Tic, self.poses_stamps[frame_id] - self.poses_stamps[self.frames[kf_idx].frame_id]

    def solve_GN_calib(self,use_calib_this_file = False):
        print("solve_GN_calib!!!!")
        
        fix_noise = 1e-6

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]
        img_size = self.frames.last_keyframe().img.shape[-2:]
        height, width = img_size

        K = self.K
        pin = self.cfg["pin"]
        pin = 0
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf <= pin:
            return
        
        

        pin = max(unique_kf_idx[-1].item()-self.window_num,0)
        print('[INFO] marg',time.time())
        if pin > self.last_pin:
            print('Marginalization!!!',pin,self.last_pin)
            # Marginalization
            marg_graph = gtsam.NonlinearFactorGraph()
            Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx[self.last_pin:])
            img_size = self.frames.last_keyframe().img.shape[-2:]
            Xs = constrain_points_to_ray(img_size, Xs, K)
            ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

            for iiii in range(self.last_pin,pin):
                self.frames_to_save.append(iiii)
            marg_mask = torch.logical_and(torch.logical_and(torch.logical_and(ii >= self.last_pin,jj>=self.last_pin),torch.logical_or(ii < pin,jj<pin)),
                                                                              torch.logical_and(ii <= self.last_pin+3,jj <= self.last_pin+3))
            print(ii,jj)
            ii = ii[marg_mask]
            jj = jj[marg_mask]
            print(ii,jj)
            idx_ii2jj = idx_ii2jj[marg_mask]
            valid_match = valid_match[marg_mask]
            Q_ii2jj = Q_ii2jj[marg_mask]
            newest =torch.max(torch.max(ii),torch.max(jj))

            new_pin = pin
            pin = self.last_pin

            if not(self.marg_factor is None):
                ssss = keys2str(self.marg_factor.keys())
                for sss in ssss:
                    if 'X' in sss and int(sss[1:])+pin > newest:
                        print(int(sss[1:])+pin)
                        newest = int(sss[1:])+pin
            
            T_WCs = T_WCs[:newest-pin+1]
            Xs = Xs[:newest-pin+1]
            Cs = Cs[:newest-pin+1]

            pose_data = T_WCs.data[:, 0, :]
            pose_data_new = getPosesRel(np.arange(pin,pin+pose_data.shape[0]),pose_data,self.wTcs,self.ss,self.enable_ms)
            aligncore = mast3r_fusion_backends.AlignCoreCalib()
            aligncore.init(
                pose_data_new,
                Xs,
                Cs,
                K,
                ii, # edge
                jj, # edge
                idx_ii2jj, # matching
                valid_match, # mask
                Q_ii2jj, # uncertainty
                height,
                width,
                pixel_border,
                z_eps,
                sigma_pixel,
                sigma_depth,
                C_thresh,
                Q_thresh,
                max_iter,
                delta_thresh,
                self.subpixel_factor,self.d_diff_threshold
            )

            H = torch.zeros([(pose_data.shape[0])*7,(pose_data.shape[0])*7],dtype=torch.float64,device='cpu')
            v = torch.zeros([(pose_data.shape[0])*7],dtype=torch.float64,device='cpu')

            H11 = torch.zeros([1,ii.shape[0],7,7],dtype=torch.float64,device='cpu')
            v11 = torch.zeros([1,ii.shape[0],7],dtype=torch.float64,device='cpu')
            c11 = torch.zeros([ii.shape[0]],dtype=torch.float64,device='cpu')
            aligncore.hessian_pieces(H11,v11,c11)
            vfactors = Align2GTSAM_factors(H11.numpy(),v11.numpy(),self.wTcs[pin:],self.ss[pin:],ii.cpu().numpy(),jj.cpu().numpy(),pin)
            for iii in range(H11.shape[1]):
                self.all_factors.append({'type':'visual','H':H11[0,iii],'v':v11[0,iii],'iijj':[ii[iii].item(),jj[iii].item()],
                                         'tstamps':[self.poses_stamps[self.frames[ii[iii]].frame_id],self.poses_stamps[self.frames[jj[iii]].frame_id]],
                                         'params':[self.wTcs[ii[iii]],self.ss[ii[iii]],self.wTcs[jj[iii]],self.ss[jj[iii]]]})
                self.all_factors.append({'type':'param','ii':ii[iii].item(),'v':self.vs[ii[iii]],'b':self.bs[ii[iii]]})
                            
            initials = gtsam.Values()
            marg_graph = gtsam.NonlinearFactorGraph()
            symbols = []
            keys_to_marg = []
            
            prior_factors = []
            for iii in range(0,T_WCs.shape[0]):
                initials.insert(X(iii),gtsam.Pose3(self.wTcs[iii+pin]))
                initials.insert(S(iii),self.ss[iii+pin])
                symbols.append(S(iii))
                symbols.append(X(iii))

                initials.insert(C(iii),gtsam.Pose3(self.Tic))
                initials.insert(Z(iii),gtsam.Pose3(self.wTcs[iii+pin] @ np.linalg.inv(self.Tic)))
                initials.insert(B(iii),self.bs[iii+pin])
                initials.insert(V(iii),self.vs[iii+pin])
                if iii+pin == 0: # Constrained initial  biases contribute to better consistency.
                    prior_factors.append(gtsam.PriorFactorConstantBias(B(iii), gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])), gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_noise)))

                if iii+pin < new_pin:
                    keys_to_marg.append(C(iii)); keys_to_marg.append(V(iii))
                    keys_to_marg.append(Z(iii)); keys_to_marg.append(X(iii))
                    keys_to_marg.append(B(iii)); keys_to_marg.append(S(iii))

                if iii > 0:
                    new_preintegration =  gtsam.PreintegratedCombinedMeasurements(self.params,self.bs[iii-1+pin])
                    dd = self.imu_pool.get_records(self.poses_stamps[self.frames[iii-1+torch.min(ii).item()].frame_id],
                                                   self.poses_stamps[self.frames[iii+torch.min(ii).item()].frame_id])
                    is_bad = False
                    for t0, t1, ddd in dd:
                        if t1 - t0 > 0.1: is_bad = True;print(t0,t1-t0,'!!!!!!!!!!!!!!!!!!!!!!!!')
                    if is_bad: new_preintegration =  gtsam.PreintegratedCombinedMeasurements(self.params_loose,self.bs[iii-1+pin])
                    for t0, t1, ddd in dd:
                        new_preintegration.integrateMeasurement(ddd[3:6],ddd[0:3]/180*math.pi,t1-t0)
                    ff = gtsam.gtsam.CombinedImuFactor(\
                                Z(iii-1),V(iii-1),Z(iii),V(iii),B(iii-1),B(iii),\
                                new_preintegration)
                    prior_factors.append(ff)

                prior_factors.append(gtsam_unstable.ExPoseConstraintFactor(Z(iii),X(iii),C(iii), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]))))

                if self.enable_excalib:
                    if iii + pin == 0:
                        prior_factors.append(gtsam.PriorFactorPose3(C(iii),gtsam.Pose3(self.Tic), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1])*0.1)))
                else:
                    prior_factors.append(gtsam.PriorFactorPose3(C(iii),gtsam.Pose3(self.Tic), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]))))
                if iii > 0 and self.enable_excalib:
                    prior_factors.append(gtsam.BetweenFactorPose3(C(iii), C(iii-1), gtsam.Pose3(np.eye(4,4)), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1])*fix_noise)))
                if iii ==0 and pin == 0:
                    if not self.enable_ms:
                        prior_factors.append(gtsam.PriorFactorPose3(Z(iii),gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1e-6,1e-6,1e-6,1e-6]))))
                    else:
                        TTT = np.eye(4,4)
                        prior_factors.append(gtsam.PriorFactorPose3(Z(iii),gtsam.Pose3(TTT), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1e-6,1e-6,1e-6,1e-6]))))
                
                # Regularization
                if iii == 0 and self.enable_ms:
                    prior_factors.append(gtsam.PriorFactorPose3(X(iii),gtsam.Pose3(self.wTcs[iii+pin]), gtsam.noiseModel.Diagonal.Sigmas(self.regularization_noise)))
                    prior_factors.append(gtsam.PriorFactorDouble(S(iii),self.ss[iii+pin], gtsam.noiseModel.Diagonal.Sigmas([1.0])))
                elif iii == 0: # just fix old frames for visual-only stage
                    prior_factors.append(gtsam.PriorFactorPose3(X(iii),gtsam.Pose3(self.wTcs[iii+pin]), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.0001,0.0001,0.0001,0.01,0.01,0.01]))))
                    prior_factors.append(gtsam.PriorFactorDouble(S(iii),self.ss[iii+pin], gtsam.noiseModel.Diagonal.Sigmas([0.0001])))
            for h_factor in vfactors:
                marg_graph.add(h_factor)
            for factor in prior_factors: 
                marg_graph.add(factor)
            if not(self.marg_factor is None): 
                marg_graph.add(self.marg_factor)
            self.marg_factor = gtsam.marginalizeOut(marg_graph,initials,keys_to_marg)
            self.marg_factor = self.marg_factor.rekey((np.array(self.marg_factor.keys())-(new_pin-pin)).tolist())
            del aligncore
            pin = new_pin
            self.last_pin = pin
        print('[INFO] marg.',time.time())

        # pin = 0
        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx[pin:])

        img_size = self.frames.last_keyframe().img.shape[-2:]

        # Constrain points to ray
        #! Calibration is needed in current version
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        mask = torch.logical_and(ii >= pin, jj >= pin)
        ii = ii[mask]
        jj = jj[mask]
        idx_ii2jj = idx_ii2jj[mask]
        valid_match = valid_match[mask]
        Q_ii2jj = Q_ii2jj[mask]
        pose_data = T_WCs.data[:, 0, :]
        assert((torch.max(ii)-torch.min(ii)).item() == pose_data.shape[0]-1)

        H = torch.zeros([(pose_data.shape[0])*7,(pose_data.shape[0])*7],dtype=torch.float64,device='cpu')
        v = torch.zeros([(pose_data.shape[0])*7],dtype=torch.float64,device='cpu')
        print('[INFO] before optim.',time.time())
        params = gtsam.LevenbergMarquardtParams();params.setMaxIterations(2)
        prior_factors = []

        T_WCs64 = lietorch.Sim3(T_WCs.data.to(torch.float64))
        while len(self.bs) < T_WCs.shape[0] + pin:
            iii = len(self.bs) - pin
            self.bs.append(gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])))
            self.vs.append(np.array([.0,.0,.0]))
            T_WC = T_WCs64[iii,0].matrix().cpu().numpy()
            T_WC[0:3,0:3] /= T_WCs64[iii,0].data[-1].item()
            self.wTcs.append(T_WC)
            self.ss.append(T_WCs64[iii,0].data[-1].item())

        aligncore = mast3r_fusion_backends.AlignCoreCalib()
        for i in range(self.cfg['max_iters']):
            # print('time1',time.time())
            pose_data_new = getPosesRel(np.arange(pin,pin+pose_data.shape[0]),pose_data,self.wTcs,self.ss,self.enable_ms)
            aligncore.init(
                pose_data_new,
                Xs,
                Cs,
                K,
                ii, # edge
                jj, # edge
                idx_ii2jj, # matching
                valid_match, # mask
                Q_ii2jj, # uncertainty
                height,
                width,
                pixel_border,
                z_eps,
                sigma_pixel,
                sigma_depth,
                C_thresh,
                Q_thresh,
                max_iter,
                delta_thresh,
                self.subpixel_factor,self.d_diff_threshold
            )

            # print('time2',time.time())
            
            H11 = torch.zeros([1,ii.shape[0],7,7],dtype=torch.float64,device='cpu')
            v11 = torch.zeros([1,ii.shape[0],7],dtype=torch.float64,device='cpu')
            c11 = torch.zeros([ii.shape[0]],dtype=torch.float64,device='cpu')
            aligncore.hessian_pieces(H11,v11,c11)
            vfactors = Align2GTSAM_factors(H11.numpy(),v11.numpy(),self.wTcs[pin:],self.ss[pin:],ii.cpu().numpy(),jj.cpu().numpy(),pin)

            initials = gtsam.Values()
            cur_graph = gtsam.NonlinearFactorGraph()
            symbols = []

            # print('1',time.time())
            for iii in range(0,T_WCs.shape[0]):
                initials.insert(X(iii),gtsam.Pose3(self.wTcs[iii+pin]))
                initials.insert(S(iii),self.ss[iii+pin])
                symbols.append(S(iii))
                symbols.append(X(iii))
                # print('c',time.time())

                if not self.enable_ms:
                    if i == 0:
                        if iii == 0:
                            prior_factors.append(gtsam.PriorFactorPose3(X(iii),gtsam.Pose3(self.wTcs[iii+pin]), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1])*0.0001)))
                            if T_WCs.shape[0]== 2:
                                prior_factors.append(gtsam.PriorFactorDouble(S(iii),3.0, gtsam.noiseModel.Diagonal.Sigmas([0.0001])))
                            else:
                                prior_factors.append(gtsam.PriorFactorDouble(S(iii),self.ss[iii+pin], gtsam.noiseModel.Diagonal.Sigmas([0.0001])))
                else:
                    initials.insert(C(iii),gtsam.Pose3(self.Tic))
                    initials.insert(Z(iii),gtsam.Pose3(self.wTcs[iii+pin] @ np.linalg.inv(self.Tic)))
                    initials.insert(B(iii),self.bs[iii+pin])
                    initials.insert(V(iii),self.vs[iii+pin])

                    if iii+pin == 0:
                        if self.enable_ms:
                            prior_factors.append(gtsam.PriorFactorConstantBias(B(iii), gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])), gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_noise)))
                        else:
                            prior_factors.append(gtsam.PriorFactorConstantBias(B(iii), gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])), gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_noise)))

                    # The prior factors are constructed at the first iteration (i==0)
                    if i == 0:
                        if iii == 0 and pin>0 :
                            prior_factors.append(self.marg_factor)

                        # print('z',time.time())
                        if iii > 0:
                            new_preintegration =  gtsam.PreintegratedCombinedMeasurements(self.params,self.bs[iii-1+pin])
                            dd = self.imu_pool.get_records(self.poses_stamps[self.frames[iii-1+torch.min(ii).item()].frame_id],
                                                           self.poses_stamps[self.frames[iii+torch.min(ii).item()].frame_id])
                            is_bad = False
                            for t0, t1, ddd in dd:
                                if t1 - t0 > 0.1: is_bad = True;print(t0,t1-t0,'!!!!!!!!!!!!!!!!!!!!!!!!')
                            if is_bad: new_preintegration =  gtsam.PreintegratedCombinedMeasurements(self.params_loose,self.bs[iii-1+pin])
                            for t0, t1, ddd in dd:
                                new_preintegration.integrateMeasurement(ddd[3:6],ddd[0:3]/180*math.pi,t1-t0)
                            ff = gtsam.gtsam.CombinedImuFactor(\
                                        Z(iii-1),V(iii-1),Z(iii),V(iii),B(iii-1),B(iii),\
                                        new_preintegration)
                            prior_factors.append(ff)

                        # Extrinsic constraint (i -> c)
                        prior_factors.append(gtsam_unstable.ExPoseConstraintFactor(Z(iii),X(iii),C(iii), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]))))

                        # Extrinsic constraint (prioir)
                        if self.enable_excalib:
                            if iii + pin == 0:
                                prior_factors.append(gtsam.PriorFactorPose3(C(iii),gtsam.Pose3(self.Tic), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1])*0.1)))
                        else:
                            prior_factors.append(gtsam.PriorFactorPose3(C(iii),gtsam.Pose3(self.Tic), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]))))
                        if iii > 0 and self.enable_excalib:
                            prior_factors.append(gtsam.BetweenFactorPose3(C(iii), C(iii-1), gtsam.Pose3(np.eye(4,4)), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1])*fix_noise)))

                        if iii ==0 and pin == 0:
                            if not self.enable_ms:
                                prior_factors.append(gtsam.PriorFactorPose3(Z(iii),gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1e-6,1e-6,1e-6,1e-6]))))
                            else:
                                TTT = np.eye(4,4)
                                # TTT[0:3,3] += 1000.0 
                                prior_factors.append(gtsam.PriorFactorPose3(Z(iii),gtsam.Pose3(TTT), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1e-6,1e-6,1e-6,1e-6]))))
                        # prior_factors.append(gtsam.PriorFactorConstantBias(B(iii),\
                        #                     gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])), \
                        #                     gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))))
            # print(time.time())
            # print('3',time.time())
            # print('time3',time.time())

            # Visual constraint
            for h_factor in vfactors:
                cur_graph.add(h_factor)
            for factor in prior_factors:
                cur_graph.add(factor)
            # print(time.time())
            # print('time4',time.time())
            
            # params.setVerbosityLM("SUMMARY")
            # if i> 5: params.setMaxIterations(20)
            optimizer = gtsam.LevenbergMarquardtOptimizer(cur_graph, initials, params)
            cur_result = optimizer.optimize()
            # print('time5',time.time())

            # print(cur_result)
            assert(T_WCs.shape[0] == pose_data.shape[0])

            for iii in range(0,T_WCs.shape[0]):
                if self.enable_ms:
                    self.Tic = cur_result.atPose3(C(0)).matrix()
                    self.bs[iii+pin] = cur_result.atConstantBias(B(iii))
                    self.vs[iii+pin] = cur_result.atVector(V(iii))
                self.ss[iii+pin] = cur_result.atDouble(S(iii))
                self.wTcs[iii+pin] = cur_result.atPose3(X(iii)).matrix()
                # print(self.bs[iii],self.vs[iii])
            self.cur_graph = cur_graph
            self.cur_result = cur_result
            pose_data_new = getPoses(np.arange(pin,pin+pose_data.shape[0]),pose_data,self.wTcs,self.ss)
            pose_data[:,:] = pose_data_new[:,:]
            
        print('[INFO] after optim.',time.time())
        print(keys2str(initials.keys()))

        # Update the keyframe T_WC
        self.frames.update_T_WCs(T_WCs, unique_kf_idx[pin:])

        if T_WCs.shape[0] == 7:
            self.solve_VI_init()





    def solve_VI_init(self):
        """ initialize the V-I system, referring to VIN-Fusion """

        pin = 0
        unique_kf_idx = self.get_unique_kf_idx()
        Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx[pin:])

        preintegrations = []
        for iii in range(0,T_WCs.shape[0]):
            if iii > 0:
                new_preintegration =  gtsam.PreintegratedCombinedMeasurements(self.params,self.bs[iii-1+pin])
                dd = self.imu_pool.get_records(self.poses_stamps[self.frames[iii-1+pin].frame_id],
                                               self.poses_stamps[self.frames[iii+pin].frame_id])
                for t0, t1, ddd in dd:
                    new_preintegration.integrateMeasurement(ddd[3:6],ddd[0:3]/180*math.pi,t1-t0)
                preintegrations.append(new_preintegration)

        sum_g = np.zeros(3,dtype = np.float64)
        ccount = 0
        for iii in range(0,T_WCs.shape[0]-1):
            dt = preintegrations[iii].deltaTij()
            tmp_g = preintegrations[iii].deltaVij()/dt
            sum_g += tmp_g
            ccount += 1

        aver_g = sum_g * 1.0 / ccount
        var_g = 0.0
        for iii in range(0,T_WCs.shape[0]-1):
            dt = preintegrations[iii].deltaTij()
            tmp_g = preintegrations[iii].deltaVij()/dt
            var_g += np.linalg.norm(tmp_g - aver_g)**2
        var_g =math.sqrt(var_g/ccount)
        if var_g < 0.0:
            print("IMU excitation not enough!")
        else:
            wTcs = []
            for iii in range(0,T_WCs.shape[0]):
                T_WC = T_WCs[iii,0].matrix().cpu().numpy()
                T_WC[0:3,0:3] /= T_WCs[iii,0].data[-1].item()
                wTcs.append(T_WC)
            vi_result = VisualIMUAlignment(self.Tic, np.array(wTcs), preintegrations, ignore_lever= True)
            print(vi_result)
            all_cs = []
            for iii in range(0,T_WCs.shape[0]):
                print(vi_result['wTbs'][iii])
                T_temp = vi_result['wTbs'][iii] @ self.Tic
                # T_temp[0:3,3] += 1000.0
                dd = np.concatenate([T_temp[0:3,3],Rotation.from_matrix(T_temp[0:3,0:3]).as_quat(),np.array([T_WCs[iii,0].data[-1].item() * vi_result['s']])])
                all_cs.append(torch.tensor(dd[None].astype(np.float32),device='cuda'))
            T_WCs = lietorch.Sim3(torch.stack(all_cs))
            self.frames.update_T_WCs(T_WCs, unique_kf_idx[pin:])

            T_WCs64 = lietorch.Sim3(T_WCs.data.to(torch.float64))
            for iii in range(T_WCs64.shape[0]):
                T_WC = T_WCs64[iii,0].matrix().cpu().numpy()
                T_WC[0:3,0:3] /= T_WCs64[iii,0].data[-1].item()
                self.wTcs[iii] = T_WC
                self.ss[iii] = T_WCs64[iii,0].data[-1].item()
                self.bs[iii] = vi_result['bs'][iii]
            self.init_vi_signal = True
            self.enable_ms = True
            print(vi_result)

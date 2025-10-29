import numpy as np
import gtsam
import gtsam_unstable
import mast3r_fusion.geoFunc.trans as trans
import mast3r_fusion.geoFunc.data_utils as data_utils
from gtsam.symbol_shorthand import B, V, X, S, Z, C
import torch

def keys2str(keys):
    ss = []
    for k in keys:
        if k >= B(0) and k< B(100000): ss.append("B"+str(k-B(0)))
        if k >= V(0) and k< V(100000): ss.append("V"+str(k-V(0)))
        if k >= X(0) and k< X(100000): ss.append("X"+str(k-X(0)))
        if k >= S(0) and k< S(100000): ss.append("S"+str(k-S(0)))
        if k >= Z(0) and k< Z(100000): ss.append("Z"+str(k-Z(0)))
        if k >= C(0) and k< C(100000): ss.append("C"+str(k-C(0)))
    return ss

def skew_sym(xx):
    x = xx[0]; y = xx[1]; z = xx[2]
    return np.array([0, -z, y, z, 0, -x, -y, x, 0]).reshape([3,3])

def VisualIMUAlignment(Tic, wTcs, preintegrations, ignore_lever = False, disable_scale = False):
    vs = []
    bs = []
    if not ignore_lever:
        wTbs = np.matmul(wTcs,np.linalg.inv(Tic))
    else:
        T_tmp = np.linalg.inv(Tic)
        T_tmp[0:3,3] = 0.0
        wTbs = np.matmul(wTcs,T_tmp)
    cost = 0.0
    # solveGyroscopeBias
    A = np.zeros([3,3])
    b = np.zeros(3)
    H1 =np.zeros([15,6], order='F', dtype=np.float64)
    H2 =np.zeros([15,3], order='F', dtype=np.float64)
    H3 =np.zeros([15,6], order='F', dtype=np.float64)
    H4 =np.zeros([15,3], order='F', dtype=np.float64)
    H5 =np.zeros([15,6], order='F', dtype=np.float64) # navstate wrt. bias
    H6 =np.zeros([15,6], order='F', dtype=np.float64)
    for i in range(0,wTcs.shape[0]):
        bs.append(gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])))
        vs.append(np.array([.0,.0,.0]))
    
    for i in range(0,wTcs.shape[0]-1):
        pose_i = gtsam.Pose3(wTbs[i])
        pose_j = gtsam.Pose3(wTbs[i+1])
        Rij = np.matmul(pose_i.rotation().matrix().T,pose_j.rotation().matrix())
        imu_factor = gtsam.gtsam.CombinedImuFactor(0,1,2,3,4,5,preintegrations[i])
        err = imu_factor.evaluateErrorCustom(pose_i,vs[i],\
                                             pose_j,vs[i+1],\
                                             bs[i],bs[i+1],\
                                             H1,H2,H3,H4,H5,H6)
        tmp_A = H5[0:3,3:6]
        tmp_b = err[0:3]
        cost +=  np.dot(tmp_b,tmp_b)
        A += np.matmul(tmp_A.T,tmp_A)
        b += np.matmul(tmp_A.T,tmp_b)
    bg = -np.matmul(np.linalg.inv(A),b)
    for i in range(0,wTcs.shape[0]):
        bs[i] = gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),bg)
    print('bg: ',bg)
    
    # linearAlignment
    all_frame_count = wTcs.shape[0]
    n_state = all_frame_count * 3 + 3 + 1
    A = np.zeros([n_state,n_state])
    b = np.zeros(n_state)
    i_count = 0
    for i in range(0,wTcs.shape[0]-1):
        pose_i = gtsam.Pose3(wTbs[i])
        pose_j = gtsam.Pose3(wTbs[i+1])
        R_i = pose_i.rotation().matrix()
        t_i = pose_i.translation()
        R_j = pose_j.rotation().matrix()
        t_j = pose_j.translation()
        pim = preintegrations[i]
        tmp_A = np.zeros([6,10])
        tmp_b = np.zeros(6)
        dt = pim.deltaTij()
        tmp_A[0:3,0:3] = -dt * np.eye(3,3)
        tmp_A[0:3,6:9] = R_i.T * dt * dt / 2
        tmp_A[0:3,9] = np.matmul(R_i.T, t_j-t_i) / 100.0
        tmp_b[0:3] = pim.deltaPij()
        tmp_A[3:6,0:3] = -np.eye(3,3)
        tmp_A[3:6,3:6] = np.matmul(R_i.T, R_j)
        tmp_A[3:6,6:9] = R_i.T * dt
        tmp_b[3:6] = pim.deltaVij()
        r_A = np.matmul(tmp_A.T,tmp_A)
        r_b = np.matmul(tmp_A.T,tmp_b)
        A[i_count*3:i_count*3+6,i_count*3:i_count*3+6] += r_A[0:6,0:6]
        b[i_count*3:i_count*3+6] += r_b[0:6]
        A[-4:,-4:] += r_A[-4:,-4:]
        b[-4:] += r_b[-4:]
        
        A[i_count*3:i_count*3+6,n_state-4:] += r_A[0:6,-4:]
        A[n_state-4:,i_count*3:i_count*3+6] += r_A[-4:,0:6]
        i_count += 1
    
    A = A * 1000.0
    b = b * 1000.0
    x = np.matmul(np.linalg.inv(A),b)
    s = x[n_state-1] / 100.0
    g = x[-4:-1]
    print('init g:',g)
    # RefineGravity
    g0 = g / np.linalg.norm(g) * 9.81
    lx = np.zeros(3)
    ly = np.zeros(3)
    n_state = all_frame_count * 3 + 2 + 1
    A = np.zeros([n_state,n_state])
    b = np.zeros(n_state)
    for k in range(4):
        aa = g / np.linalg.norm(g)
        tmp = np.array([.0,.0,1.0])
        bb = (tmp - np.dot(aa,tmp) * aa)
        bb /= np.linalg.norm(bb)
        cc = np.cross(aa,bb)
        bc = np.zeros([3,2])
        bc[0:3,0] = bb
        bc[0:3,1] = cc
        lxly = bc
        
        i_count = 0
        for i in range(0,wTcs.shape[0]-1):
            pose_i = gtsam.Pose3(wTbs[i])
            pose_j = gtsam.Pose3(wTbs[i+1])
            R_i = pose_i.rotation().matrix()
            t_i = pose_i.translation()
            R_j = pose_j.rotation().matrix()
            t_j = pose_j.translation()
            tmp_A = np.zeros([6,9])
            tmp_b = np.zeros(6)
            pim = preintegrations[i]
            dt = pim.deltaTij()
            tmp_A[0:3,0:3] = -dt *np.eye(3,3)
            tmp_A[0:3,6:8] = np.matmul(R_i.T,lxly) * dt * dt /2 
            tmp_A[0:3,8]   = np.matmul(R_i.T,t_j - t_i) / 100.0
            tmp_b[0:3] = pim.deltaPij() - np.matmul(R_i.T,g0) * dt * dt / 2
            tmp_A[3:6,0:3] = -np.eye(3)
            tmp_A[3:6,3:6] = np.matmul(R_i.T,R_j)
            tmp_A[3:6,6:8] = np.matmul(R_i.T,lxly) * dt
            tmp_b[3:6] = pim.deltaVij() - np.matmul(R_i.T,g0) * dt
            r_A = np.matmul(tmp_A.T,tmp_A)
            r_b = np.matmul(tmp_A.T,tmp_b)
            A[i_count*3:i_count*3+6,i_count*3:i_count*3+6] += r_A[0:6,0:6]
            b[i_count*3:i_count*3+6] += r_b[0:6]
            A[-3:,-3:] += r_A[-3:,-3:]
            b[-3:] += r_b[-3:]
            A[i_count*3:i_count*3+6,n_state-3:] += r_A[0:6,-3:]
            A[n_state-3:,i_count*3:i_count*3+6] += r_A[-3:,0:6]
            i_count += 1
        
        A = A * 1000.0
        b = b * 1000.0
        x = np.matmul(np.linalg.inv(A),b)
        dg = x[-3:-1]
        g0 = g0 + np.matmul(lxly,dg)
        g0 = g0 / np.linalg.norm(g0) * 9.81
        s = x[-1] / 100.0
        # print(s,g0,x)
    if disable_scale:
        s = 1.0
        
    print('g,s:',g,s)
    if np.fabs(np.linalg.norm(g) - 9.81 ) < 0.5 and s > 0:
        print('V-I successfully initialized!')
    
    # visualInitialAlign
    wTbs[:,0:3,3] *= s # !!!!!!!!!!!!!!!!!!!!!!!!
    for i in range(0, wTcs.shape[0]):
        vs[i] = np.matmul(wTbs[i,0:3,0:3],x[i*3:i*3+3])
    
    # g2R
    ng1 = g0/ np.linalg.norm(g0)
    ng2 = np.array([0,0,1.0])
    R0 = trans.FromTwoVectors(ng1,ng2)
    yaw = trans.R2ypr(R0)[0]
    R0 = np.matmul(trans.ypr2R(np.array([-yaw,0,0])),R0)
    g = np.matmul(R0,g0)
    for i in range(0,wTcs.shape[0]):
        wTbs[i,0:3,3] = np.matmul(R0,wTbs[i,0:3,3])
        wTbs[i,0:3,0:3] = np.matmul(R0,wTbs[i,0:3,0:3])
        vs[i] = np.matmul(R0, vs[i])
    return {"wTbs":wTbs,"vs":vs,"bs":bs,"s":s}

def coarse_calib_torch(img_size, Xs, max_iter=10000, eps=1e-6, lr=1e-5):
    torch.set_grad_enabled(True)
    device = Xs.device
    N = Xs.shape[0]
    h, w = img_size

    u, v = torch.meshgrid(torch.arange(w), torch.arange(h), indexing="xy")
    uv = torch.stack((u, v), dim=-1).unsqueeze(0).view(*Xs.shape[:-1], 2)
    uv = uv.to('cuda').to(torch.float32)
    f = nn.Parameter(torch.tensor([300.0, 300.0, w / 2, h / 2], dtype=torch.float32, device=device))
    optimizer = optim.SGD([f], lr=lr,momentum=0.5)

    prev_loss = float("inf")

    for it in tqdm(range(max_iter), desc="Calib training"):
        fx, fy, cx, cy = f
        k1, k2, p1, p2  = [0,0,0,0]
        residuals = torch.tensor(0.0, device=device)
        count = 0

        h, w = img_size

        u, v = uv[:, 0], uv[:, 1]

        # 掩码: 只保留中间区域
        mask = (u >= w * 0.1) & (u <= w * 0.9) & (v >= h * 0.1) & (v <= h * 0.9)
        Xs = Xs[mask]
        uv = uv[mask]

        x = Xs[:, 0] / Xs[:, 2]
        y = Xs[:, 1] / Xs[:, 2]
        x2 = x * x
        y2 = y * y
        r2 = x2 + y2
        r4 = r2 * r2

        # 畸变
        xd = x * (1 + k1 * r2/100 + k2 * r4/100) + 2 * p1 * x * y/100 + p2 * (r2 + 2 * x2)/100
        yd = y * (1 + k1 * r2/100 + k2 * r4/100) + p1 * (r2 + 2 * y2)/100 + 2 * p2 * x * y/100

        u_est = fx * xd + cx
        v_est = fy * yd + cy

        # 重投影误差
        res_u = uv[:, 0] - u_est
        res_v = uv[:, 1] - v_est
        residuals = res_u ** 2 + res_v ** 2

        loss = residuals.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # 打印详细信息
        tqdm.write(f"Iter {it}: loss={loss.item():.4e}, fx={fx.item():.2f}, fy={fy.item():.2f}, "
                   f"cx={cx.item():.2f}, cy={cy.item():.2f} ")

        if abs(prev_loss - loss.item()) < eps:
            print(f"收敛于第 {it + 1} 次迭代, loss={loss.item():.4e}")
            break

        prev_loss = loss.item()
    torch.set_grad_enabled(False)
    return f.detach().cpu().numpy()




    # def batch_calib(self,ii,jj,idx_ii2jj,valid_match,Q_ii2jj):
    #     fp_log_converg = open('converg.txt','wt')
    #     C_thresh = self.cfg["C_conf"]
    #     Q_thresh = self.cfg["Q_conf"]
    #     pixel_border = self.cfg["pixel_border"]
    #     z_eps = self.cfg["depth_eps"]
    #     max_iter = self.cfg["max_iters"]
    #     sigma_pixel = self.cfg["sigma_pixel"]
    #     sigma_depth = self.cfg["sigma_depth"]
    #     delta_thresh = self.cfg["delta_norm"]
    #     img_size = self.frames.last_keyframe().img.shape[-2:]
    #     height, width = img_size
    #     K = self.K
    #     unique_kf_idx = torch.unique(torch.cat([ii, jj]), sorted=True)

    #     img_size = self.frames.last_keyframe().img.shape[-2:]
    #     h, w = img_size
    #     fix_noise = 1e-6

    #     # uv = uv.to('cuda').to(torch.float32)

        
    #     for iiiter in range(1000):
    #         torch.set_grad_enabled(True)
    #         f = nn.Parameter(torch.tensor([K[0,0], w / 2, h / 2], dtype=torch.float32, device='cuda'))
    #         optimizer = optim.Adam([f], lr=0.1)
    #         Xs_o, T_WCs_o, Cs = self.get_poses_points(unique_kf_idx)
    #         if iiiter%2 == 0:
    #             Xs = constrain_points_to_ray(img_size, Xs_o, K)
    #             pkl_data = {'pts':[],'poses':[],'clr':[]}
    #             for idx in range(Xs.shape[0]):
    #                 Xsw = T_WCs_o[idx] * Xs[idx]
    #                 pkl_data['pts'].append(Xsw.cpu().numpy())
    #                 pkl_data['poses'].append(T_WCs_o[idx].data.cpu().numpy())
    #                 pkl_data['clr'].append(self.frames[idx].uimg.cpu().numpy())
    #             pickle.dump(pkl_data,open('temp/%010d.pkl'%iiiter,'wb'))
    #         for iiter in range(10):
    #             residuals = torch.tensor(0.0, device=f.device)
    #             K = torch.zeros(3, 3, device=f.device)
    #             K[0, 0] = f[0]  # fx
    #             K[1, 1] = f[0]  # fy
    #             K[0, 2] = f[1]  # cx
    #             K[1, 2] = f[2]  # cy
    #             K[2, 2] = 1.0   # 固定为1
    #             Xs = constrain_points_to_ray(img_size, Xs_o, K)
    #             for idx in range(ii.shape[0]):
    #                 i = ii[idx]
    #                 j = jj[idx]
    #                 T_WCs_i = T_WCs_o[i]
    #                 T_WCs_j = T_WCs_o[j]
    #                 Tij = T_WCs_i.inv() * T_WCs_j
    #                 Xsn = Tij * Xs[j]
    #                 u_est = Xsn[:,0] / Xsn[:,2] * f[0] + f[1]
    #                 v_est = Xsn[:,1] / Xsn[:,2] * f[0] + f[2]
    #                 up = idx_ii2jj[idx] % w
    #                 vp = idx_ii2jj[idx] // w
    #                 residuals += torch.sum(((up-u_est)*valid_match[idx,:,0])**2)
    #                 residuals += torch.sum(((vp-v_est)*valid_match[idx,:,0])**2)
    #             residuals = residuals/h/w
    #             loss = residuals.sum()
    #             loss.backward()
    #             optimizer.step()
    #             optimizer.zero_grad()
    #             print(f"Iter {iiter}: loss={loss.item():.4e}, {str(f)}")
    #         torch.set_grad_enabled(False)
    #         K = torch.zeros(3, 3, device=f.device)
    #         K[0, 0] = f[0]  # fx
    #         K[1, 1] = f[0]  # fy
    #         K[0, 2] = f[1]  # cx
    #         K[1, 2] = f[2]  # cy
    #         K[2, 2] = 1.0   # 固定为1
    #         fp_log_converg.writelines('%.3f %.3f %.3f\n'%(f[0].item(),f[1].item(),f[2].item()))
    #         fp_log_converg.flush()
    #         pin = 0
    #         Xs, T_WCs, Cs = self.get_poses_points(unique_kf_idx[pin:])
    #         img_size = self.frames.last_keyframe().img.shape[-2:]
    #         Xs = constrain_points_to_ray(img_size, Xs, K)
    #         ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()
    #         mask = torch.logical_and(ii >= pin, jj >= pin)
    #         ii = ii[mask]
    #         jj = jj[mask]
    #         idx_ii2jj = idx_ii2jj[mask]
    #         valid_match = valid_match[mask]
    #         Q_ii2jj = Q_ii2jj[mask]
    #         pose_data = T_WCs.data[:, 0, :]
    #         assert((torch.max(ii)-torch.min(ii)).item() == pose_data.shape[0]-1)
    #         H = torch.zeros([(pose_data.shape[0])*7,(pose_data.shape[0])*7],dtype=torch.float64,device='cpu')
    #         v = torch.zeros([(pose_data.shape[0])*7],dtype=torch.float64,device='cpu')
    #         print('before optim.',time.time())
    #         prior_factors = []

    #         T_WCs64 = lietorch.Sim3(T_WCs.data.to(torch.float64))
    #         while len(self.bs) < T_WCs.shape[0] + pin:
    #             iii = len(self.bs) - pin
    #             self.bs.append(gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])))
    #             self.vs.append(np.array([.0,.0,.0]))
    #             T_WC = T_WCs64[iii,0].matrix().cpu().numpy()
    #             T_WC[0:3,0:3] /= T_WCs64[iii,0].data[-1].item()
    #             self.wTcs.append(T_WC)
    #             self.ss.append(T_WCs64[iii,0].data[-1].item())

    #         for i in range(10):
    #             print('time1',time.time())
    #             pose_data_new = getPosesRel(np.arange(pin,pin+pose_data.shape[0]),pose_data,self.wTcs,self.ss,self.enable_ms)
    #             aligncore = mast3r_fusion_backends.AlignCoreCalib()
    #             aligncore.init(
    #                 pose_data_new,
    #                 Xs,
    #                 Cs,
    #                 K,
    #                 ii, # edge
    #                 jj, # edge
    #                 idx_ii2jj, # matching
    #                 valid_match, # mask
    #                 Q_ii2jj, # uncertainty
    #                 height,
    #                 width,
    #                 pixel_border,
    #                 z_eps,
    #                 sigma_pixel,
    #                 sigma_depth,
    #                 C_thresh,
    #                 Q_thresh,
    #                 max_iter,
    #                 delta_thresh,
    #                 self.subpixel_factor, self.d_diff_threshold
    #             )
    #             print('time2',time.time())

    #             H11 = torch.zeros([1,ii.shape[0],7,7],dtype=torch.float64,device='cpu')
    #             v11 = torch.zeros([1,ii.shape[0],7],dtype=torch.float64,device='cpu')
    #             c11 = torch.zeros([ii.shape[0]],dtype=torch.float64,device='cpu')
    #             aligncore.hessian_pieces(H11,v11,c11)
    #             vfactors = Align2GTSAM_factors(H11.numpy(),v11.numpy(),self.wTcs[pin:],self.ss[pin:],ii.cpu().numpy(),jj.cpu().numpy(),pin)
    #             initials = gtsam.Values()
    #             cur_graph = gtsam.NonlinearFactorGraph()
    #             symbols = []

    #             print('time3',time.time())
    #             for iii in range(0,T_WCs.shape[0]):
    #                 initials.insert(X(iii),gtsam.Pose3(self.wTcs[iii+pin]))
    #                 initials.insert(S(iii),self.ss[iii+pin])
    #                 symbols.append(S(iii))
    #                 symbols.append(X(iii))
    #                 # print('c',time.time())

    #                 if not self.enable_ms:
    #                     if i == 0:
    #                         if iii == 0:
    #                             prior_factors.append(gtsam.PriorFactorPose3(X(iii),gtsam.Pose3(self.wTcs[iii+pin]), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1])*0.0001)))
    #                             if T_WCs.shape[0]== 2:
    #                                 prior_factors.append(gtsam.PriorFactorDouble(S(iii),3.0, gtsam.noiseModel.Diagonal.Sigmas([0.0001])))
    #                             else:
    #                                 prior_factors.append(gtsam.PriorFactorDouble(S(iii),self.ss[iii+pin], gtsam.noiseModel.Diagonal.Sigmas([0.0001])))
    #                 else:
    #                     initials.insert(C(iii),gtsam.Pose3(self.Tic))
    #                     initials.insert(Z(iii),gtsam.Pose3(self.wTcs[iii+pin] @ np.linalg.inv(self.Tic)))
    #                     initials.insert(B(iii),self.bs[iii+pin])
    #                     initials.insert(V(iii),self.vs[iii+pin])
    #                     # prior_factors.append(gtsam.PriorFactorDouble(S(iii),self.ss[iii+pin], gtsam.noiseModel.Diagonal.Sigmas([0.01])))

    #                     if iii+pin == 0:
    #                         if self.enable_ms:
    #                             prior_factors.append(gtsam.PriorFactorConstantBias(B(iii), gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])), gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_noise)))
    #                         else:
    #                             prior_factors.append(gtsam.PriorFactorConstantBias(B(iii), gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])), gtsam.noiseModel.Diagonal.Sigmas(self.init_bias_noise)))

    #                     if i == 0:
    #                         if iii == 0 and pin>0 :
    #                             prior_factors.append(self.marg_factor)

    #                         if iii > 0:
    #                             new_preintegration =  gtsam.PreintegratedCombinedMeasurements(self.params,self.bs[iii-1+pin])
    #                             dd = self.imu_pool.get_records(self.poses_stamps[self.frames[iii-1+torch.min(ii).item()].frame_id],
    #                                                            self.poses_stamps[self.frames[iii+torch.min(ii).item()].frame_id])
    #                             is_bad = False
    #                             for t0, t1, ddd in dd:
    #                                 if t1 - t0 > 0.1: is_bad = True;print(t0,t1-t0,'!!!!!!!!!!!!!!!!!!!!!!!!')
    #                             if is_bad: new_preintegration =  gtsam.PreintegratedCombinedMeasurements(self.params_loose,self.bs[iii-1+pin])
    #                             for t0, t1, ddd in dd:
    #                                 new_preintegration.integrateMeasurement(ddd[3:6],ddd[0:3]/180*math.pi,t1-t0)
    #                             ff = gtsam.gtsam.CombinedImuFactor(\
    #                                         Z(iii-1),V(iii-1),Z(iii),V(iii),B(iii-1),B(iii),\
    #                                         new_preintegration)
    #                             prior_factors.append(ff)

    #                         # print('z',time.time())

    #                         # Extrinsic constraint (i -> c)
    #                         prior_factors.append(gtsam_unstable.ExPoseConstraintFactor(Z(iii),X(iii),C(iii), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1])*fix_noise)))

    #                         # Extrinsic constraint (prioir)
    #                         prior_factors.append(gtsam.PriorFactorPose3(C(iii),gtsam.Pose3(self.Tic), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1])*fix_noise)))

    #                         # Extrinsic constraint (temporal)
    #                         # if iii > 0:
    #                         #     prior_factors.append(gtsam.BetweenFactorPose3(C(iii), C(iii-1), gtsam.Pose3(np.eye(4,4)), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1,1,1,1])*0.0001)))

    #                         if iii ==0 and pin == 0:
    #                             if not self.enable_ms:
    #                                 prior_factors.append(gtsam.PriorFactorPose3(Z(iii),gtsam.Pose3(), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1e-6,1e-6,1e-6,1e-6]))))
    #                             else:
    #                                 TTT = np.eye(4,4)
    #                                 prior_factors.append(gtsam.PriorFactorPose3(Z(iii),gtsam.Pose3(TTT), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1e-6,1e-6,1e-6,1e-6]))))
    #             # Visual constraint
    #             for h_factor in vfactors:
    #                 cur_graph.add(h_factor)
    #             for factor in prior_factors:
    #                 cur_graph.add(factor)
    #             # print(time.time())
    #             print('time4',time.time())

    #             params = gtsam.LevenbergMarquardtParams();params.setMaxIterations(2)
    #             optimizer = gtsam.LevenbergMarquardtOptimizer(cur_graph, initials, params)
    #             cur_result = optimizer.optimize()
    #             # print(cur_result)
    #             assert(T_WCs.shape[0] == pose_data.shape[0])

    #             for iii in range(0,T_WCs.shape[0]):
    #                 if self.enable_ms:
    #                     self.Tic = cur_result.atPose3(C(0)).matrix()
    #                     self.bs[iii+pin] = cur_result.atConstantBias(B(iii))
    #                     self.vs[iii+pin] = cur_result.atVector(V(iii))
    #                 self.ss[iii+pin] = cur_result.atDouble(S(iii))
    #                 self.wTcs[iii+pin] = cur_result.atPose3(X(iii)).matrix()
    #                 # print(self.bs[iii],self.vs[iii])
    #             pose_data_new = getPoses(np.arange(pin,pin+pose_data.shape[0]),pose_data,self.wTcs,self.ss)
    #             pose_data[:,:] = pose_data_new[:,:]
    #             print('time5',time.time())

    #         print('after optim.',time.time())
    #         print(keys2str(initials.keys()))
    #         for iii in range(T_WCs.shape[0]-1,T_WCs.shape[0]):
    #             bb = self.bs[iii+torch.min(ii).item()].vector()
    #             self.fp.writelines('%.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f\n' % (self.poses_stamps[self.frames[iii+torch.min(ii).item()].frame_id],
    #                                                                                  pose_data[iii][0].item(),
    #                                                                                  pose_data[iii][1].item(),
    #                                                                                  pose_data[iii][2].item(),
    #                                                                                  pose_data[iii][3].item(),
    #                                                                                  pose_data[iii][4].item(),
    #                                                                                  pose_data[iii][5].item(),
    #                                                                                  pose_data[iii][6].item(),
    #                                                                                  pose_data[iii][7].item(),
    #                                                                                  bb[0],bb[1],bb[2],
    #                                                                                  bb[3],bb[4],bb[5]))
    #             self.fp.flush()

    #         # Update the keyframe T_WC
    #         self.frames.update_T_WCs(T_WCs, unique_kf_idx[pin:])

import matplotlib.pyplot as plt
import pickle
import numpy as np
import gtsam
import gtsam_unstable
from gtsam.symbol_shorthand import B, V, X, S, Z, C, M, G
import mast3r_fusion.geoFunc.data_utils as data_utils
import math
import yaml
from scipy.spatial.transform import Rotation
import torch
import mast3r_fusion.geoFunc.trans as trans
import lietorch
import bisect
import time
import os
import argparse
from matplotlib import cm

def skew_sym(xx):
    x = xx[0]; y = xx[1]; z = xx[2]
    return np.array([0, -z, y, z, 0, -x, -y, x, 0]).reshape([3,3])

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

def CustomHessianFactor(symbols, values, H: np.ndarray, v: np.ndarray):
    info_expand = np.zeros([H.shape[0]+1,H.shape[1]+1])
    info_expand[0:-1,0:-1] = H
    info_expand[0:-1,-1] = v
    info_expand[-1,-1] = 1000000.0 # This is meaningless.
    dims = []
    for sym in symbols:
        if sym - X(0) < 100000 and sym >= X(0):
            dims.append(6)
        if sym - S(0) < 100000 and sym >= S(0):
            dims.append(1)
    h_f = gtsam.HessianFactor(symbols,dims,info_expand)
    l_c = gtsam.LinearContainerFactor(h_f,values)
    return l_c

def Align2GTSAM_factors(H11: np.ndarray, v11: np.ndarray, lin_list, wTcs, ss, ii, jj, pin):
    factors = []
    # i0 = np.min(np.concatenate([ii,jj]))
    # assert(i0==pin)
    for idx in range(ii.shape[0]):
        i = ii[idx] - pin
        j = jj[idx] - pin

        # correct ddx
        wTc0 = lin_list[idx][0] 
        s0 = lin_list[idx][1] 
        wTc1 = lin_list[idx][2] 
        s1 = lin_list[idx][3] 
        dd0 = np.concatenate([wTc0[0:3,3],Rotation.from_matrix(wTc0[0:3,0:3]).as_quat(),np.array([s0])])
        X0 = lietorch.Sim3(torch.tensor(dd0.astype(np.float64)))
        dd1 = np.concatenate([wTc1[0:3,3],Rotation.from_matrix(wTc1[0:3,0:3]).as_quat(),np.array([s1])])
        X1 = lietorch.Sim3(torch.tensor(dd1.astype(np.float64)))
        dx_lin = (X0.inv()*X1).log()

        wTc0 = wTcs[i]
        s0 = ss[i]
        wTc1 = wTcs[j]
        s1 = ss[j]
        dd0 = np.concatenate([wTc0[0:3,3],Rotation.from_matrix(wTc0[0:3,0:3]).as_quat(),np.array([s0])])
        X0 = lietorch.Sim3(torch.tensor(dd0.astype(np.float64)))
        dd1 = np.concatenate([wTc1[0:3,3],Rotation.from_matrix(wTc1[0:3,0:3]).as_quat(),np.array([s1])])
        X1 = lietorch.Sim3(torch.tensor(dd1.astype(np.float64)))
        dx_cur = (X0.inv()*X1).log()
        ddx = (dx_cur - dx_lin).numpy()

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
        v = v11[0,idx,:] + H @ ddx
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNSS-IMU-LIO Integration")

    parser.add_argument("--config", type=str, default="config/base_kitti360.yaml", help="config file")
    parser.add_argument("--graph_path", type=str, default="graph_33000.pkl",
                        help="Path to the input graph pickle file")
    parser.add_argument("--loop_path", type=str, default="graph_loop_33000.pkl",
                        help="Path to the loop closure graph pickle file")
    parser.add_argument("--calib_path", type=str, default="config/intrinsics_0412.yaml",
                        help="Path to calibration file")
    parser.add_argument("--imu_path", type=str, default="/mnt/d/Data/0412_full/adis_imu.txt",
                        help="Path to IMU data")
    parser.add_argument("--imu_dt", type=float, default=0.00,
                        help="IMU time offset")
    parser.add_argument("--enable_gnss", action="store_true",
                        help="Enable GNSS usage (set flag to activate)")
    parser.add_argument("--gnss_path", type=str)
    parser.add_argument("--result_path", type=str, default="graph_33000_gnss.pkl.txt",
                        help="Path to save result")

    args = parser.parse_args()

    print("Graph path:", args.graph_path)
    print("Loop graph path:", args.loop_path)
    print("Calibration:", args.calib_path)
    print("IMU path:", args.imu_path)
    print("IMU dt:", args.imu_dt)
    print("Enable GNSS:", args.enable_gnss)
    print("Result path:", args.result_path)

    GRAPH_PATH = args.graph_path
    loop_path = args.loop_path
    CALIB_PATH = args.calib_path
    CONFIG_PATH = args.config
    IMU_PATH = args.imu_path
    imu_dt = args.imu_dt
    ENABLE_GNSS = args.enable_gnss
    RESULT_PATH = args.result_path


    all_gnss = np.array([])
    if ENABLE_GNSS:
        all_gnss = np.loadtxt(args.gnss_path)
        LEVER = np.array([0.020, 0.285, 0.423]) - np.array([0.0,0.47,-0.04])

    config = yaml.load(open(CONFIG_PATH,'rt'), Loader=yaml.SafeLoader)
    calib = yaml.load(open(CALIB_PATH,'rt'), Loader=yaml.SafeLoader)
    Tic = np.copy(calib['Tic'])

    if config['ms_opt']['imu_format'] == 'custom_deg':
        try:
            imu_pool = data_utils.IMUPool(np.loadtxt(args.imu_path,delimiter=' '), degree = True, dt = imu_dt)
        except:
            imu_pool = data_utils.IMUPool(np.loadtxt(args.imu_path,delimiter=','), degree = True, dt = imu_dt)
    elif config['ms_opt']['imu_format'] == 'custom_rad':
        try:
            imu_pool = data_utils.IMUPool(np.loadtxt(args.imu_path,delimiter=' '), degree = False, dt = imu_dt)
        except:
            imu_pool = data_utils.IMUPool(np.loadtxt(args.imu_path,delimiter=','), degree = False, dt = imu_dt)
    elif config['ms_opt']['imu_format'] == 'subt':
        all_imu = np.loadtxt(args.imu_path,delimiter=',',comments='#',skiprows=1)
        all_imu[:,0] /= 1e9
        all_imu_new = np.zeros_like(all_imu)
        all_imu_new[:,0] = all_imu[:,0]
        all_imu_new[:,1:4] = all_imu[:,5:8] * 180/math.pi
        all_imu_new[:,4:7] = all_imu[:,8:11]
        all_imu_new = all_imu_new[:,:7]
        imu_pool = data_utils.IMUPool(all_imu_new, degree = True, dt = imu_dt)
    else:
        raise Exception()
    
    noise = np.array(config['global_opt']['imu_noise'])
    

    # Notice that we have 3 sets of IMU params!
    # one for initialization of the graph
    # one for common cases
    # one for bad IMU cases
    accel_noise_sigma = noise[0] * 1
    gyro_noise_sigma = noise[1]  * 1
    accel_bias_rw_sigma = noise[2] * 1
    gyro_bias_rw_sigma = noise[3] * 1
    GRAVITY = 9.81
    measured_acc_cov = np.eye(3,3) * math.pow(accel_noise_sigma,2)
    measured_omega_cov = np.eye(3,3) * math.pow(gyro_noise_sigma,2)
    integration_error_cov = np.eye(3,3) * 0e-8
    bias_acc_cov = np.eye(3,3) * math.pow(accel_bias_rw_sigma,2)
    bias_omega_cov = np.eye(3,3) * math.pow(gyro_bias_rw_sigma,2)
    bias_acc_omega_init = np.eye(6,6) * 0e-5
    params_init = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
    params_init.setAccelerometerCovariance(measured_acc_cov)
    params_init.setIntegrationCovariance(integration_error_cov)
    params_init.setGyroscopeCovariance(measured_omega_cov)
    params_init.setBiasAccCovariance(bias_acc_cov)
    params_init.setBiasOmegaCovariance(bias_omega_cov)
    params_init.setBiasAccOmegaInit(bias_acc_omega_init)


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

    accel_noise_sigma = noise[0] * 1
    gyro_noise_sigma =  noise[1] * 1
    accel_bias_rw_sigma = noise[2]
    gyro_bias_rw_sigma = noise[3]
    GRAVITY = 9.81
    measured_acc_cov = np.eye(3,3) * math.pow(accel_noise_sigma,2)
    measured_omega_cov = np.eye(3,3) * math.pow(gyro_noise_sigma,2)
    integration_error_cov = np.eye(3,3) * 0e-8
    bias_acc_cov = np.eye(3,3) * math.pow(accel_bias_rw_sigma,2)
    bias_omega_cov = np.eye(3,3) * math.pow(gyro_bias_rw_sigma,2)
    bias_acc_omega_init = np.eye(6,6) * 0e-5
    params_loose = gtsam.PreintegrationCombinedParams.MakeSharedU(GRAVITY)
    params_loose.setAccelerometerCovariance(measured_acc_cov)
    params_loose.setIntegrationCovariance(integration_error_cov)
    params_loose.setGyroscopeCovariance(measured_omega_cov)
    params_loose.setBiasAccCovariance(bias_acc_cov)
    params_loose.setBiasOmegaCovariance(bias_omega_cov)
    params_loose.setBiasAccOmegaInit(bias_acc_omega_init)

    

    all_poses = {}    
    all_ss = {}
    all_vs = {}
    all_bs = {}
    all_t = {}

    t_list = []
    ii_list = []
    jj_list = []
    H_list = []
    v_list = []
    lin_list = []
    

    all_loops = []
    all_loop_dd = []
    count = 0

    dd = pickle.load(open(GRAPH_PATH,'rb'))
    for ddd in dd:
        if ddd['type'] == 'visual':
            all_poses[ddd['iijj'][0]] = ddd['params'][0]
            all_poses[ddd['iijj'][1]] = ddd['params'][2]
            all_ss[ddd['iijj'][0]] = ddd['params'][1]
            all_ss[ddd['iijj'][1]] = ddd['params'][3]
            all_t[ddd['iijj'][0]] = ddd['tstamps'][0]
            all_t[ddd['iijj'][1]] = ddd['tstamps'][1]   
            ii_list.append(ddd['iijj'][0])   
            jj_list.append(ddd['iijj'][1])
            H_list.append(ddd['H'])                                                                                 
            v_list.append(ddd['v'])
            lin_list.append([ddd['params'][0],ddd['params'][1],ddd['params'][2],ddd['params'][3]])
            count += 1
        elif ddd['type'] == 'param':
            all_vs[ddd['ii']] = ddd['v']
            all_bs[ddd['ii']] = ddd['b']

    if os.path.exists(loop_path):
        dd = pickle.load(open(loop_path,'rb'))
        for ddd in dd:
            if ddd['type'] == 'visual_loop':
                all_loops.append(ddd['iijj'])
                all_loop_dd.append(ddd)


    wTcs_list =[]
    ss_list = []
    bs_list = []
    vs_list = []
    for iii in sorted(all_poses.keys()):
        wTcs_list.append(all_poses[iii])
        ss_list.append(all_ss[iii])
        try:
            vs_list.append(all_vs[iii])
        except:
            vs_list.append(np.array([.0,.0,.0]))
        try:
            bs_list.append(all_bs[iii])
        except:
            bs_list.append(gtsam.imuBias.ConstantBias(np.array([.0,.0,.0]),np.array([.0,.0,.0])))
    H_list = np.array(H_list)
    v_list = np.array(v_list)
    wTcs_list = np.array(wTcs_list)
    ss_list = np.array(ss_list)
    ii_list = np.array(ii_list)
    jj_list = np.array(jj_list)
    t_list = np.array(sorted(all_t.values()))

    #! GNSS alignment (optional)
    if len(all_gnss) < 1:
        xyz_ref = np.array([0,0,0])
        dT = np.eye(4,4)
    else:
        all_gnss = all_gnss[all_gnss[:,0]>t_list[0]]
        all_gnss = all_gnss[all_gnss[:,0]<t_list[-1]]
        # GNSS alignment
        pos_list_local = []
        pos_list_global = []
        xyz_ref = None
        dist = 0.0
        dT = None
        for i in range(all_gnss.shape[0]):
            tt = all_gnss[i,0]
            idx = bisect.bisect(t_list,tt-0.001) - 1
            if idx < 0 or idx > len(t_list) - 2: continue
            if xyz_ref is None: 
                xyz_ref = all_gnss[i,1:4]
            dd = imu_pool.get_records(t_list[idx],tt)
            iii = idx
            new_preintegration =  gtsam.PreintegratedCombinedMeasurements(params,bs_list[iii])
            for t0, t1, ddd in dd:
                new_preintegration.integrateMeasurement(ddd[3:6],ddd[0:3]/180*np.pi,t1-t0)
            T_pred = new_preintegration.predict(gtsam.NavState(gtsam.Pose3(wTcs_list[iii] @ np.linalg.inv(Tic)),vs_list[iii]),bs_list[iii]).pose().matrix()
            pos_list_local.append(T_pred[0:3,3])
            pos_global = np.array(trans.cart2enu(xyz_ref,all_gnss[i,1:4] - xyz_ref))
            pos_list_global.append(pos_global)
            if len(pos_list_local)>1:
                dist += np.linalg.norm(pos_list_local[-1][0:2] - pos_list_local[0][0:2])
            if dist > 10 and dT is None:
                dT = np.eye(4,4)
                dxyz0 = pos_list_local[-1] - pos_list_local[0]
                dxyz1 = pos_list_global[-1] - pos_list_global[0]
                dyaw = np.arctan2(dxyz0[1],dxyz0[0]) - np.arctan2(dxyz1[1],dxyz1[0])
                dR = trans.att2m([0,0,-dyaw])
                dT[0:3,0:3] = dR
                dT[0:3,3] = pos_list_global[0] - dR @   pos_list_local[0] 
                break


    for i in range(len(wTcs_list)):
        wTcs_list[i] = dT @ wTcs_list[i]
        vs_list[i] = dT[0:3,0:3] @ vs_list[i]

    
    CCCCC = len(wTcs_list)

    for iiter in range(6):
        initials = gtsam.Values()
        cur_graph = gtsam.NonlinearFactorGraph()

        #! Add visual factors
        vfactors = Align2GTSAM_factors(H_list[None],v_list[None],lin_list, wTcs_list,ss_list,ii_list,jj_list,0)
        for vf in vfactors:
            cur_graph.add(vf)

        #! Add relative factors
        loop_i = []
        for iii in range(0,CCCCC):
            initials.insert(X(iii),gtsam.Pose3(wTcs_list[iii]))
            initials.insert(S(iii),ss_list[iii])
            initials.insert(B(iii),bs_list[iii])
            initials.insert(V(iii),vs_list[iii])
            initials.insert(Z(iii),gtsam.Pose3(wTcs_list[iii] @ np.linalg.inv(Tic)))
        initials.insert(C(0),gtsam.Pose3(Tic))

        #! Add IMU factors, extrinsic factors
        prior_factors = []
        prior_factors.append(gtsam.PriorFactorPose3(C(0),gtsam.Pose3(Tic), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]))))
        for iii in range(0,CCCCC):
            if iii == 0 and dT is None:
                TTT= np.eye(4,4)
                prior_factors.append(gtsam.PriorFactorPose3(Z(iii),gtsam.Pose3(TTT), gtsam.noiseModel.Diagonal.Sigmas(np.array([1,1,1e-6,1e-6,1e-6,1e-6]))))
            prior_factors.append(gtsam_unstable.ExPoseConstraintFactor(Z(iii),X(iii),C(0), gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-4,1e-4,1e-4,1e-4,1e-4,1e-4]))))
            if iii > 0:
                dd = imu_pool.get_records(all_t[iii-1],all_t[iii])
                if iiter < 3:
                    new_preintegration =  gtsam.PreintegratedCombinedMeasurements(params_init,bs_list[iii-1])
                else:
                    new_preintegration =  gtsam.PreintegratedCombinedMeasurements(params,bs_list[iii-1])
                is_bad = False
                for t0, t1, ddd in dd:
                    if t1 - t0 > 0.1: is_bad = True;print(t0,t1-t0,'!!!!!!!!!!!!!!!!!!!!!!!!')
                if is_bad: 
                    new_preintegration =  gtsam.PreintegratedCombinedMeasurements(params_loose,bs_list[iii-1])
                    # quit()
                for t0, t1, ddd in dd:
                    new_preintegration.integrateMeasurement(ddd[3:6],ddd[0:3]/180*np.pi,t1-t0)
                ff = gtsam.gtsam.CombinedImuFactor(\
                            Z(iii-1),V(iii-1),Z(iii),V(iii),B(iii-1),B(iii),\
                            new_preintegration)
                prior_factors.append(ff)# print(new_preintegration)
    
        for pf in prior_factors:
            cur_graph.add(pf)

        

        #! Add loop factors
        ii_loop_list = []
        jj_loop_list = []
        H_loop_list = []
        v_loop_list = []
        lin_loop_list = []
        if iiter > 1:
            for i in range(0,len(all_loops)):
                wTc0 = all_loop_dd[i]['params'][0]
                wTc1 = all_loop_dd[i]['params'][2]
                iii = all_loop_dd[i]['iijj'][0]
                jjj = all_loop_dd[i]['iijj'][1]

                MMM = np.linalg.inv(wTc0) @ wTc1
                MMMp = np.linalg.inv(initials.atPose3(X(iii)).matrix()) @ initials.atPose3(X(jjj)).matrix()

                threshold = 100000000000000000 # accept all loops, leave them to Cauchy function
                threshold_t =100000000000000000

                if np.linalg.norm(trans.m2att(MMM[0:3,0:3]))*57.3 < threshold and np.linalg.norm(MMM[0:3,3]) < threshold_t:
                    if iiter > 1:
                        noise = gtsam.noiseModel.Robust.Create(\
                                          gtsam.noiseModel.mEstimator.Cauchy(100.0),\
                              gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001,0.001,0.001,0.01,0.01,0.01])))
                    else:
                        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001,0.001,0.001,0.01,0.01,0.01]))
                    f = gtsam_unstable.ExPoseConstraintFactor(X(iii),
                    X(jjj),M(iii*100000+jjj), gtsam.noiseModel.Diagonal.Sigmas(np.array([0.001,0.001,0.001,0.01,0.01,0.01])))
                    f1 = gtsam.PriorFactorPose3(M(iii*100000+jjj),gtsam.Pose3(np.linalg.inv(wTc0) @ wTc1), noise)

                    if iiter <=2:
                        initials.insert(M(iii*100000+jjj),gtsam.Pose3(np.linalg.inv(wTc0) @ wTc1))
                        cur_graph.add(f1)
                        cur_graph.add(f)
                    if (iiter>2):
                        dT = np.linalg.inv(MMMp) @ MMM
                        distance = np.linalg.norm(dT[0:3,3])
                        dangle = np.linalg.norm(Rotation.from_matrix(dT[0:3,0:3]).as_rotvec()) * 57.3
                        if distance > 1 or dangle > 3.0: continue
                        else:
                            if np.linalg.norm(trans.m2att(MMM[0:3,0:3]))*57.3 < 90:
                                ii_loop_list.append(all_loop_dd[i]['iijj'][0])   
                                jj_loop_list.append(all_loop_dd[i]['iijj'][1])
                                H_loop_list.append(all_loop_dd[i]['H'])                                                                                 
                                v_loop_list.append(all_loop_dd[i]['v'])
                                lin_loop_list.append([all_loop_dd[i]['params'][0],all_loop_dd[i]['params'][1],all_loop_dd[i]['params'][2],all_loop_dd[i]['params'][3]])
                            else:
                                initials.insert(M(iii*100000+jjj),gtsam.Pose3(np.linalg.inv(wTc0) @ wTc1))
                                cur_graph.add(f1)
                                cur_graph.add(f)
                    loop_i.append(i)

        if len(ii_loop_list) > 0:
            H_loop_list = np.array(H_loop_list)
            v_loop_list = np.array(v_loop_list)
            ii_loop_list = np.array(ii_loop_list)
            jj_loop_list = np.array(jj_loop_list)
            vfactors_loop = Align2GTSAM_factors(H_loop_list[None],v_loop_list[None],lin_loop_list, wTcs_list,ss_list,ii_loop_list,jj_loop_list,0)
            for vf in vfactors_loop:
                cur_graph.add(vf)

        #! Add GNSS factors
        for i in range(all_gnss.shape[0]):
            tt = all_gnss[i,0]
            pos_global = np.array(trans.cart2enu(xyz_ref,all_gnss[i,1:4] - xyz_ref))

            idx = bisect.bisect(t_list,tt-0.001) - 1
            if idx < 0: continue
            if idx > len(t_list) - 2:break
            iii = idx

            # use imu preintegration to bridge keyframe and GNSS
            # abandon too long preintegrations
            if np.fabs(tt > t_list[idx]) > 5.0 : continue
            dd = imu_pool.get_records(t_list[idx],tt)
            new_preintegration =  gtsam.PreintegratedCombinedMeasurements(params,bs_list[iii])
            for t0, t1, ddd in dd:
                new_preintegration.integrateMeasurement(ddd[3:6],ddd[0:3]/180*np.pi,t1-t0)
            ff = gtsam.gtsam.CombinedImuFactor(\
                        Z(iii),V(iii),G(i),G(i+200000),B(iii),G(i+100000),\
                        new_preintegration)
            noise = gtsam.noiseModel.Robust.Create(\
                          gtsam.noiseModel.mEstimator.Cauchy(25),\
                gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0,1.0,10.0])*0.01/4))
            if args.enable_gap:
                noise =  gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0,1.0,10.0])*0.01)
            gnss_factor = gtsam.GPSFactorLever(G(i), pos_global, LEVER,noise)
            
            vvv = new_preintegration.predict(gtsam.NavState(gtsam.Pose3(wTcs_list[iii] @ np.linalg.inv(Tic)),vs_list[iii]),bs_list[iii]).velocity()
            TTT = new_preintegration.predict(gtsam.NavState(gtsam.Pose3(wTcs_list[iii] @ np.linalg.inv(Tic)),vs_list[iii]),bs_list[iii]).pose()
            initials.insert(G(i),TTT)
            initials.insert(G(i + 100000),bs_list[iii])
            initials.insert(G(i + 200000),vvv)

            cur_graph.push_back(gnss_factor)
            cur_graph.push_back(ff)

        # let's go!
        opt_params = gtsam.LevenbergMarquardtParams()
        opt_params.setMaxIterations(20)
        opt_params.setVerbosityLM("SUMMARY") 
        optimizer = gtsam.LevenbergMarquardtOptimizer(cur_graph, initials, opt_params)
        print(cur_graph.error(initials))
        cur_result = optimizer.optimize()
        for iii in range(0,CCCCC):
            wTcs_list[iii] = cur_result.atPose3(X(iii)).matrix()
            ss_list[iii] = cur_result.atDouble(S(iii))
            bs_list[iii] = cur_result.atConstantBias(B(iii))
            vs_list[iii] = cur_result.atVector(V(iii))
        Tic = cur_result.atPose3(C(0)).matrix()

    # bs_series= []
    # for iii in range(0,wTcs_list.shape[0]):
    #     bs_series.append(bs_list[iii].vector())
    # bs_series = np.array(bs_series)

    # plt.figure('bias')
    # plt.subplot(1,2,1)
    # plt.plot(bs_series[:,0])
    # plt.plot(bs_series[:,1])
    # plt.plot(bs_series[:,2])
    # plt.subplot(1,2,2)
    # plt.plot(bs_series[:,3])
    # plt.plot(bs_series[:,4])
    # plt.plot(bs_series[:,5])
    # plt.savefig('test_bias.png')

    #! Output results
    t_series = []
    x_series = []
    y_series = []
    s_series = []
    fp_out = open(RESULT_PATH,'wt')
    # for idx in sorted(all_poses.keys()):
    for idx in range(0,CCCCC):
        wTc = cur_result.atPose3(X(idx)).matrix()
        x_series.append(wTc[0,3])
        y_series.append(wTc[1,3])
        s_series.append(all_ss[idx])
        t_series.append(all_t[idx])
        ttt = wTc[0:3,3]
        qqq = Rotation.from_matrix(wTc[0:3,0:3]).as_quat()
        bias = bs_list[idx].vector()
        fp_out.writelines('%.3f %.5f %.5f %.5f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %.10f %d %d %.10f %.10f %.10f\n' %\
         (all_t[idx],ttt[0],ttt[1],ttt[2],qqq[0],qqq[1],qqq[2],qqq[3],ss_list[idx],\
         bias[0],bias[1],bias[2],bias[3],bias[4],bias[5],idx,1,xyz_ref[0],xyz_ref[1],xyz_ref[2]))


    #! Visualize loop edges
    plt.figure('234',figsize=[20,20])
    from matplotlib import cm, colors
    base_cmap = cm.get_cmap("autumn")
    def smooth_remap(x, gamma=0.5):
        return np.power(x, gamma)

    norm = plt.Normalize(vmin=0, vmax=np.pi)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "compressed_autumn", base_cmap(smooth_remap(np.linspace(0,1,256)))
    )

    yaw_diffs = [] 
    
    for i in loop_i:
        iii = all_loop_dd[i]['iijj'][0]
        jjj = all_loop_dd[i]['iijj'][1]
        
        pose_i = cur_result.atPose3(X(iii))
        pose_j = cur_result.atPose3(X(jjj))
        
        R_i = pose_i.rotation().matrix()
        R_j = pose_j.rotation().matrix()
        yaw_i = np.arctan2(R_i[1,0], R_i[0,0])
        yaw_j = np.arctan2(R_j[1,0], R_j[0,0])
        
        yaw_diff = np.arctan2(np.sin(yaw_j - yaw_i), np.cos(yaw_j - yaw_i))
        yaw_diff_abs = np.abs(yaw_diff)
        yaw_diffs.append(yaw_diff_abs)
        
        color = new_cmap(norm(yaw_diff_abs))
        
        plt.plot([pose_i.x(), pose_j.x()],
                 [pose_i.y(), pose_j.y()],
                 c=color, zorder=10000)
    
    sm = cm.ScalarMappable(cmap=new_cmap, norm=norm)
    sm.set_array([])  
    cbar = plt.gcf().colorbar(sm, ax=plt.gca())
    cbar.set_label("Yaw difference (rad)")

    plt.plot(x_series,y_series)
    plt.axis('equal')

    plt.show()
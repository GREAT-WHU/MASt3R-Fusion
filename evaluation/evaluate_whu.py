import argparse
import logging
import typing
import yaml

import numpy as np

import evo.common_ape_rpe as common
from evo.core import lie_algebra, sync, metrics
from evo.core.result import Result
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
from evo.tools import file_interface, log
from evo.tools.settings import SETTINGS

import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation
import bisect
import math
import time
import mast3r_fusion.geoFunc.trans as trans
import mast3r_fusion.geoFunc.data_utils as data_utils

import matplotlib

logger = logging.getLogger(__name__)

SEP = "-" * 80  # separator line

def ape(traj_ref: PosePath3D, traj_est: PosePath3D,
        pose_relation: metrics.PoseRelation, align: bool = False,
        correct_scale: bool = False, n_to_align: int = -1,
        align_origin: bool = False, ref_name: str = "reference",
        est_name: str = "estimate",
        change_unit: typing.Optional[metrics.Unit] = None) -> Result:
    if n_to_align >0 : 
        print('[INFO]>> only use the starting segment')
        n_to_align = np.where((np.array(traj_ref.timestamps)[1:]-np.array(traj_ref.timestamps)[0:-1])>100)[0][0]-1

    # Align the trajectories.
    only_scale = correct_scale and not align
    alignment_transformation = None
    if align or correct_scale:
        logger.debug(SEP)
        alignment_transformation = lie_algebra.sim3(
            *traj_est.align(traj_ref, correct_scale, only_scale, n=n_to_align))
    elif align_origin:
        logger.debug(SEP)
        alignment_transformation = traj_est.align_origin(traj_ref)

    # Calculate APE.
    logger.debug(SEP)
    data = (traj_ref, traj_est)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)

    if change_unit:
        ape_metric.change_unit(change_unit)

    title = str(ape_metric)
    if align and not correct_scale:
        title += "\n(with SE(3) Umeyama alignment)"
    elif align and correct_scale:
        title += "\n(with Sim(3) Umeyama alignment)"
    elif only_scale:
        title += "\n(scale corrected)"
    elif align_origin:
        title += "\n(with origin alignment)"
    else:
        title += "\n(not aligned)"
    if (align or correct_scale) and n_to_align != -1:
        title += " (aligned poses: {})".format(n_to_align)

    ape_result = ape_metric.get_result(ref_name, est_name)
    ape_result.info["title"] = title

    logger.debug(SEP)
    logger.info(ape_result.pretty_str())

    ape_result.add_trajectory(ref_name, traj_ref)
    ape_result.add_trajectory(est_name, traj_est)
    if isinstance(traj_est, PoseTrajectory3D):
        seconds_from_start = np.array(
            [t - traj_est.timestamps[0] for t in traj_est.timestamps])
        ape_result.add_np_array("seconds_from_start", seconds_from_start)
        ape_result.add_np_array("timestamps", traj_est.timestamps)
        ape_result.add_np_array("distances_from_start", traj_ref.distances)
        ape_result.add_np_array("distances", traj_est.distances)

    if alignment_transformation is not None:
        ape_result.add_np_array("alignment_transformation_sim3",
                                alignment_transformation)

    return ape_result

def hex_to_rgb01(hex_str):
    s = hex_str.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:  # e.g. "f0a" -> "ff00aa"
        s = ''.join(ch*2 for ch in s)
    if len(s) != 6:
        raise ValueError("3 or 6 characters")

    try:
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
    except ValueError:
        raise ValueError("illegal characters")

    return [r / 255.0, g / 255.0, b / 255.0]

def load_gt(path):
    all_data_gt = {}
    ref_gt = np.loadtxt(path)
    xyz_ref = None
    Ten0 = None
    for i in range(ref_gt.shape[0]):
        tt = ref_gt[i,0]
        if xyz_ref is None: xyz_ref = ref_gt[i,1:4]
        Ren = trans.Cen(xyz_ref)
        ani = [ref_gt[i,4]/180*math.pi,\
                ref_gt[i,5]/180*math.pi,\
                ref_gt[i,6]/180*math.pi]
        Rni = trans.att2m(ani)
        Rei = np.matmul(Ren,Rni)
        tei = ref_gt[i,1:4]
        Tei = np.eye(4,4)
        Tei[0:3,0:3] = Rei
        Tei[0:3,3] = tei
        if Ten0 is None:
            Ten0 = np.eye(4,4)
            Ten0[0:3,0:3] = Ren
            Ten0[0:3,3] = tei
        Tn0i = np.matmul(np.linalg.inv(Ten0),Tei)
        TTT = Tn0i
        all_data_gt[tt] = {'T':TTT}
    return all_data_gt, xyz_ref


if __name__ == '__main__':
    color_list = [hex_to_rgb01('#1FB5B4'),hex_to_rgb01('#FF99FF'),hex_to_rgb01('#FFA500'),[1,0,0]]
    plt.figure('123')
    fig, axes = plt.subplots(1, 2,figsize=np.array([104/25.4*1.6,49/25.4*2.5]), 
                             gridspec_kw={'width_ratios': [1,1.45]}) 
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.subcommand = 'tum'
    args.pose_relation = 'trans_part'
    args.align = True
    args.correct_scale = False
    args.n_to_align = 1
    args.align_origin = False
    args.plot_mode = 'xyz'
    args.plot_x_dimension = 'seconds'
    args.plot_colormap_min = None
    args.plot_colormap_max = None
    args.plot_colormap_max_percentile = None
    args.ros_map_yaml = None
    args.plot = True
    args.est_files = ['result_whu1.txt','result_post_whu1.txt']
    args.est_files = ['result_whu2.txt','result_post_whu2.txt']
    GT_PATH = '/workspace/mast3r_fusion_open/whu_dataset/gt_whu.txt'
    CALIB_PATH = 'config/intrinsics_whu.yaml'
    label_list = ['MAST3R-Fusion','MAST3R-Fusion (post)']
    args.save_plot = False
    args.serialize_plot = False

    #! load reference poses
    all_data_gt, xyz_ref = load_gt(GT_PATH)
    fp_temp = open('gt_temp.txt','wt')
    for tt in sorted(all_data_gt.keys()):
        TTT = all_data_gt[tt]['T']
        qqq = Rotation.from_matrix(TTT[0:3,0:3]).as_quat()
        fp_temp.writelines('%.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n'%(tt,TTT[0,3],TTT[1,3],TTT[2,3],qqq[0],qqq[1],qqq[2],qqq[3]))
    fp_temp.close()
    args.ref_file = 'gt_temp.txt'
    calib = yaml.load(open(CALIB_PATH,'rt'), Loader=yaml.SafeLoader)

    Tic = np.copy(calib['Tic'])
    pred_count = 0
    first_ok = False
    for iii in range(len(args.est_files)):
        t_list = []
        s_list = []
        dd = np.loadtxt(args.est_files[iii])
        with open('result_temp.txt','wt') as f:
            start = 0
            if 'result' in args.est_files[iii]: start = 0
            if 'vins_result' in args.est_files[iii]:
                dd_vins = np.loadtxt(args.est_files[iii])
                dd_vins = dd_vins[np.fabs(dd_vins[:,0]-np.round(dd_vins[:,0]))<0.001]
                dd_vins = dd_vins[dd_vins[:,0]<282569]
                dd_vins = dd_vins[dd_vins[:,0]>281189]
                wTcs_list = []
                t_list = []
                for i in range(dd_vins.shape[0]):
                    xyz = dd_vins[i,1:4]
                    vxyz = dd_vins[i,4:7]
                    att = dd_vins[i,7:10]
                    R = trans.att2m(att/180*math.pi)
                    t_list.append(dd_vins[i,0])
                    TTT = np.eye(4,4)
                    t = xyz
                    q = Rotation.from_matrix(R).as_quat()
                    f.writelines('%f %f %f %f %f %f %f %f\n'%(dd_vins[i,0],t[0],t[1],t[2],q[0],q[1],q[2],q[3]))
            elif 'DBA' in args.est_files[iii]:
                dd_dba = np.loadtxt(args.est_files[iii])
                for i in range(dd_dba.shape[0]):
                    f.writelines('%f %f %f %f %f %f %f %f\n'%(dd_dba[i,0],dd_dba[i,1],dd_dba[i,2],dd_dba[i,3],dd_dba[i,4],dd_dba[i,5],dd_dba[i,6],dd_dba[i,7]))
            else:
                for iiii in range(start,dd.shape[0]):
                    t_list.append(dd[iiii,0])
                    if 'rtk' in args.est_files[iii]:
                        s_list.append(1)
                    else:
                        s_list.append(dd[iiii,8])
                    if 'gnss' in args.est_files[iii] :
                        enu_new = trans.cart2enu(xyz_ref,trans.enu2cart(dd[iiii][-3:],dd[iiii,1:4]) + dd[iiii][-3:] - xyz_ref)
                    elif 'rtk' in args.est_files[iii]:
                        enu_new = trans.cart2enu(xyz_ref,dd[iiii][1:4] - xyz_ref)
                    else:
                        enu_new  = dd[iiii,1:4]
                    TTT = np.eye(4,4)
                    TTT[0:3,3] = enu_new
                    if 'rtk' in args.est_files[iii]:
                        pass
                    else:
                        TTT[0:3,0:3] = Rotation.from_quat(dd[iiii,4:8]).as_matrix()
                    Twi = TTT @ np.linalg.inv(Tic)
                    # Twi = TTT
                    t = Twi[0:3,3]
                    q = Rotation.from_matrix(Twi[0:3,0:3]).as_quat()
                    if 'result' in args.est_files[iii]  and (not 'post' in args.est_files[iii]):
                        if iiii > 0 and  dd[iiii,0] < dd[iiii-1,0]:
                            first_ok = True
                        if iiii>0 and dd[iiii,0] == dd[iiii-1,0] and first_ok == True:
                            pred_count = 0
                            f.writelines('%f %f %f %f %f %f %f %f\n'%(dd[iiii,0],t[0],t[1],t[2],q[0],q[1],q[2],q[3]))
        
                        else:
                            if first_ok:
                                pred_count += 1
                                if pred_count < 10:
                                    f.writelines('%f %f %f %f %f %f %f %f\n'%(dd[iiii,0],t[0],t[1],t[2],q[0],q[1],q[2],q[3]))
                    else:
                        f.writelines('%f %f %f %f %f %f %f %f\n'%(dd[iiii,0],t[0],t[1],t[2],q[0],q[1],q[2],q[3]))

        if not ('rtk' in args.est_files[iii]) and len(args.est_files[iii]) == 1:
            plt.figure()
            plt.subplot(2,1,1)
            plt.plot(dd[:,0],dd[:,-4-3-3])
            plt.plot(dd[:,0],dd[:,-3-3-3])
            plt.plot(dd[:,0],dd[:,-2-3-3])
            plt.subplot(2,1,2)
            plt.plot(dd[:,0],dd[:,-4-3])
            plt.plot(dd[:,0],dd[:,-3-3])
            plt.plot(dd[:,0],dd[:,-2-3])
            plt.figure('s',figsize=[20,5])
            plt.plot(t_list,s_list,linewidth=0.7)
            plt.savefig('result_temp_s.png')   

        # if iii == 0:continue
            # np.savetxt(f,dd[:,0:8])
        args.est_file = 'result_temp.txt'

        if args.est_file.find('visual') != -1:
            args.correct_scale = True
        else:
            args.correct_scale = False
        traj_ref, traj_est, ref_name, est_name = common.load_trajectories(args)
        traj_ref_sel, traj_est_sel = sync.associate_trajectories(
            traj_ref, traj_est, 0.01,0.0,
            first_name=ref_name, snd_name=est_name)
        args.n_to_align = -1
        pose_relation = common.get_pose_relation(args)
        if not ('gnss' in args.est_files[iii] or 'rtk' in args.est_files[iii]):
            result = ape(traj_ref=traj_ref_sel, traj_est=traj_est_sel,
                         pose_relation=pose_relation, align=args.align,
                         correct_scale=args.correct_scale, n_to_align=args.n_to_align,
                         align_origin=args.align_origin, ref_name=ref_name,
                         est_name=est_name)
            # result = ape(traj_ref=traj_ref_sel, traj_est=traj_est_sel,
            #              pose_relation=pose_relation, align=False,
            #              correct_scale=args.correct_scale, n_to_align=args.n_to_align,
            #              align_origin=False, ref_name=ref_name,
            #              est_name=est_name)
            traj_est_sel = copy.deepcopy(result.trajectories[est_name])
            T01 = result.np_arrays['alignment_transformation_sim3']
            print(T01)
            result = ape(traj_ref=traj_ref_sel, traj_est=traj_est_sel,
                         pose_relation=pose_relation, align=args.align,
                         correct_scale=False, n_to_align=-1,
                         align_origin=args.align_origin, ref_name=ref_name,
                         est_name=est_name)
            print(result)
            traj_est.transform(T01)

        traj_ref_sel_temp = copy.deepcopy(traj_ref_sel)
        traj_est_sel_temp = copy.deepcopy(traj_est_sel)
        # traj_est_sel_temp.transform(T01)

        
        
        plt.figure('123')
        plt.sca(axes[0])

        # if iii == 0:
        x0_series=[]
        y0_series=[]
        z0_series=[]
        ax0_series=[]
        ay0_series=[]
        az0_series=[]
        
        for i in range(len(traj_ref_sel_temp.poses_se3)):
            TTT = traj_ref_sel_temp.poses_se3[i]
            x0_series.append(TTT[0,3])
            y0_series.append(TTT[1,3])
            z0_series.append(TTT[2,3])
            att = np.array(trans.m2att(TTT[0:3,0:3]))*57.3
            ax0_series.append(att[0])
            ay0_series.append(att[1])
            az0_series.append(att[2])
        if iii == 0:
            plt.plot(x0_series,y0_series,c=[0,0,0],linestyle = '--',linewidth = 0.5,zorder=1000)

        x_series=[]
        y_series=[]
        z_series=[]
        ax_series=[]
        ay_series=[]
        az_series=[]
        t_series = []
        for i in range(len(traj_est_sel_temp.poses_se3)):
            TTT = traj_est_sel_temp.poses_se3[i]
            t_series.append(traj_ref_sel_temp.timestamps[i])
            x_series.append(TTT[0,3])
            y_series.append(TTT[1,3])
            z_series.append(TTT[2,3])
            att = np.array(trans.m2att(TTT[0:3,0:3]))*57.3
            ax_series.append(att[0])
            ay_series.append(att[1])
            az_series.append(att[2])
            ppp = TTT[0:3,3]
            qqq = Rotation.from_matrix(TTT[:3, :3]/np.power(np.linalg.det(TTT[:3, :3]),1.0/3)).as_quat()
        plt.plot(x_series,y_series,c=color_list[iii],label = label_list[iii])
        errx = np.array(x_series) - np.array(x0_series)
        erry = np.array(y_series) - np.array(y0_series)
        errz = np.array(z_series) - np.array(z0_series)
        plt.gca().set_aspect(1)



        t_series=[]
        x_series=[]
        y_series=[]
        z_series=[]
        for i in range(len(traj_ref_sel.timestamps)):
            T0 = traj_ref_sel.poses_se3[i]
            T1 = traj_est_sel.poses_se3[i]
            T01 = np.matmul(np.linalg.inv(T0),T1)
            att = Rotation.from_matrix(T01[0:3,0:3]).as_rotvec()
            t_series.append(traj_ref_sel.timestamps[i])
            x_series.append(att[0]*57.3)
            y_series.append(att[1]*57.3)
            z_series.append(att[2]*57.3)



        # print('Evaluating relative pose error ...')
        subtraj_length = np.array([100,200,300,400,500,600,700,800])
        max_dist_difH = 1

        dist = traj_ref_sel.distances
        poses_ref = np.array(traj_ref_sel.poses_se3)
        poses_est = np.array(traj_est_sel.poses_se3)

        traj_len = len(dist)

        rel_trans_error_dist = []
        rel_att_error_dist = []

        for L in subtraj_length:
            max_d = 0.2 * L
            subsection_index = []

            j = 0
            k = 1
            target = dist + L

            while j < traj_len-1 and k < traj_len:
                while k < traj_len and dist[k] < target[j] - max_dist_difH:
                    k += 1
                if k < traj_len and abs(dist[k] - target[j]) <= max_dist_difH:
                    subsection_index.append((j, k))
                j += 1
                if k <= j:
                    k = j + 1

            print(f"The trajectory at {L}m has {len(subsection_index)} matching points...")

            if not subsection_index:
                rel_trans_error_dist.append(0)
                rel_att_error_dist.append(0)
                continue

            idx0 = np.array([p[0] for p in subsection_index])
            idx1 = np.array([p[1] for p in subsection_index])

            T_gt_1 = poses_ref[idx0]
            T_gt_2 = poses_ref[idx1]
            T_est_1 = poses_est[idx0]
            T_est_2 = poses_est[idx1]

            # T_gt_12 = inv(T_gt_1) * T_gt_2
            T_gt_12 = np.einsum('nij,njk->nik', np.linalg.inv(T_gt_1), T_gt_2)
            T_est_12 = np.einsum('nij,njk->nik', np.linalg.inv(T_est_1), T_est_2)
            T_err = np.einsum('nij,njk->nik', np.linalg.inv(T_gt_12), T_est_12)

            trans_err = np.linalg.norm(T_err[:, 0:3, 3], axis=1)

            rotvec = Rotation.from_matrix(T_err[:, 0:3, 0:3]).as_rotvec()
            att_err = np.linalg.norm(rotvec, axis=1)

            rel_trans_error_dist.append(np.mean(trans_err / L * 100))
            rel_att_error_dist.append(np.mean(att_err / L * 100 / np.pi * 180))

        print('Relative Translation Error: %.6f %%' % np.mean(rel_trans_error_dist))
        print('Relative Rotation Error: %.6f deg / 100 m' % np.mean(rel_att_error_dist))

    plt.savefig('whu.svg')
    plt.show()
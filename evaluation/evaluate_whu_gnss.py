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
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

SEP = "-" * 80  # separator line

def ape(traj_ref: PosePath3D, traj_est: PosePath3D,
        pose_relation: metrics.PoseRelation, align: bool = False,
        correct_scale: bool = False, n_to_align: int = -1,
        align_origin: bool = False, ref_name: str = "reference",
        est_name: str = "estimate",
        change_unit: typing.Optional[metrics.Unit] = None) -> Result:
    if n_to_align >0 : 
        print('>>>>> only use the starting segment')
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
    color_list = [[0.8,0.8,0.8],hex_to_rgb01('#1FB5B4'),hex_to_rgb01('#FF99FF'),hex_to_rgb01('#FF9500'),[1,0,0]]

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
    
    args.est_files = ['/workspace/mast3r_fusion_open/whu_dataset/whu1/rtk.txt','result_post_gnss_whu1.txt']
    GT_PATH = '/workspace/mast3r_fusion_open/whu_dataset/whu1/gt.txt'
    # args.est_files = ['/workspace/mast3r_fusion_open/whu_dataset/whu2/rtk.txt','result_post_gnss_whu2.txt']
    # GT_PATH = '/workspace/mast3r_fusion_open/whu_dataset/whu2/gt.txt'
    CALIB_PATH = 'config/intrinsics_whu.yaml'
    label_list = ['RTK','MAST3R-Fusion']
    args.save_plot = False
    args.serialize_plot = False
    t0 = None


    #! load reference poses
    all_data_gt, xyz_ref = load_gt(GT_PATH)
    args.ref_file = 'gt_temp.txt'
    calib = yaml.load(open(CALIB_PATH,'rt'), Loader=yaml.SafeLoader)
    Tic = np.copy(calib['Tic'])
    for iii in range(len(args.est_files)):
        print(args.est_files[iii])
        if 'rtk' in args.est_files[iii]:
            fp_temp = open('gt_temp.txt','wt')
            LEVER = np.copy(calib['lever'])
            Tia = np.eye(4,4)
            Tia[0:3,3] = LEVER
            for tt in sorted(all_data_gt.keys()):
                TTT = all_data_gt[tt]['T'] @ Tia
                qqq = Rotation.from_matrix(TTT[0:3,0:3]).as_quat()
                fp_temp.writelines('%.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n'%(tt,TTT[0,3],TTT[1,3],TTT[2,3],qqq[0],qqq[1],qqq[2],qqq[3]))
            fp_temp.close()
        else:
            fp_temp = open('gt_temp.txt','wt')
            for tt in sorted(all_data_gt.keys()):
                TTT = all_data_gt[tt]['T']
                qqq = Rotation.from_matrix(TTT[0:3,0:3]).as_quat()
                fp_temp.writelines('%.6f %.6f %.6f %.6f %.8f %.8f %.8f %.8f\n'%(tt,TTT[0,3],TTT[1,3],TTT[2,3],qqq[0],qqq[1],qqq[2],qqq[3]))
            fp_temp.close()
        
        t_list = []
        s_list = []
        for i_skip in range(1000):
            try:
                dd = np.loadtxt(args.est_files[iii],skiprows=i_skip)
                break
            except:
                continue
        # dd = np.loadtxt(args.est_files[iii])
        with open('result_temp.txt','wt') as f:
            start = 0
            if 'result' in args.est_files[iii]: start = 0 
            for iiii in range(start,dd.shape[0]):
                t_list.append(dd[iiii,0])
                if 'rtk' in args.est_files[iii]:
                    enu_new = trans.cart2enu(xyz_ref,dd[iiii][1:4] - xyz_ref)
                else:
                    enu_new = trans.cart2enu(xyz_ref,trans.enu2cart(dd[iiii][-3:],dd[iiii,1:4]) + dd[iiii][-3:] - xyz_ref)
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
                f.writelines('%f %f %f %f %f %f %f %f\n'%(dd[iiii,0],t[0],t[1],t[2],q[0],q[1],q[2],q[3]))


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

        traj_ref_sel_temp = copy.deepcopy(traj_ref_sel)
        traj_est_sel_temp = copy.deepcopy(traj_est_sel)
        # traj_est_sel_temp.transform(T01)

        
        plt.figure('123',figsize=np.array([104/25.4*1.6,49/25.4*1]))
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

        x_series=[]
        y_series=[]
        z_series=[]
        ax_series=[]
        ay_series=[]
        az_series=[]
        t_series = []
        for i in range(len(traj_est_sel_temp.poses_se3)):
            TTT = traj_est_sel_temp.poses_se3[i]
            t_series.append(traj_est_sel_temp.timestamps[i])
            if t0 is None:
                t0 = t_series[-1]
            x_series.append(TTT[0,3])
            y_series.append(TTT[1,3])
            z_series.append(TTT[2,3])
            att = np.array(trans.m2att(TTT[0:3,0:3]))*57.3
            ax_series.append(att[0])
            ay_series.append(att[1])
            az_series.append(att[2])
            ppp = TTT[0:3,3]
            qqq = Rotation.from_matrix(TTT[:3, :3]/np.power(np.linalg.det(TTT[:3, :3]),1.0/3)).as_quat()
        # plt.plot(x_series,y_series,c=color_list[iii],label = label_list[iii])
        errx = np.array(x_series) - np.array(x0_series)
        erry = np.array(y_series) - np.array(y0_series)
        err2 = errx**2+erry**2
        mask = err2 < 2500
        rmse = np.sqrt(np.sum(err2[mask])/np.sum(mask))
        t_series = np.array(t_series)[mask]
        ax_series = np.array(ax_series)[mask]
        ay_series = np.array(ay_series)[mask]
        az_series = np.array(az_series)[mask]
        ax0_series = np.array(ax0_series)[mask]
        ay0_series = np.array(ay0_series)[mask]
        az0_series = np.array(az0_series)[mask]
        errx = errx[mask]
        erry = erry[mask]
        err2 = err2[mask]
        plt.plot(np.array(t_series)-t0,np.sqrt(np.array(errx)**2 + np.array(erry)**2),c=color_list[iii],label='%s $\mathbf{%.3f}$'%(args.est_files[iii],rmse))
        t_series = np.array(t_series)
        plt.xlim([0,t_series[-1]-t0]);plt.ylim([0,20])
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(100))
        print('%30s   Horiz. RMSE: %.3f'%(args.est_files[iii],rmse))
    lg = plt.legend(loc='upper right',markerscale=3,fontsize=8,framealpha=1,ncol=1,columnspacing=0.3,handletextpad=0.3,edgecolor='black',fancybox=False)
    lg.set_zorder(200)
    lg.get_frame().set_linewidth(1)

    plt.savefig('result_temp.svg')
    plt.savefig('result_temp.png')

    plt.show()

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

    plt.show()

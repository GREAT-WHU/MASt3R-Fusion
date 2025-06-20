# MASt3R-Fusion

>High-Functionality Multi-Sensor SLAM without Bundle Adjustment.



[Coming soon...]

## What is this? 



MASt3R-Fusion is a SLAM system that tightly integrates feed-forward pointmap regression with multi-sensor data (e.g., IMU, GNSS), drawing inspiration from MASt3R-SLAM. It is designed for practical, large-scale 3D perception and mapping applications, and offers the following key features:

- ✅ High accuracy, high robustness, metric scale
- ✅ Full-view depth
- ✅ Multi-sensor fusion support
- ✅ Loop closure
- ✅ Real-time sliding-window tracking + globally consistent optimization
- ✅ Auto calibration  

## Preview 

<div style="display: flex; justify-content: center; gap: 20px;" align=center>
  <img alt="" src="assets/mast3r-fusion1.gif" width="400px" />
  <img alt="" src="assets/mast3r-fusion2.gif" width="400px" />
</div>

## Evaluation (Early-Stage)

<p align="center">
  <img src="./assets/0002.svg" alt="图像描述" width="600px"><br>
  <em>Fig. 1 Monocular VIO performance on KITTI-360 0002.<br>
  (Using evo-toolkit with global SE(3) alignment, no loop closure.)</em>
</p>


<p align="center">
  <img src="./assets/0002_plt.svg" alt="图像描述" width="1200px"><br>
  <em>Fig. 2 Comparison with MASt3R-SLAM on KITTI-360 0002.</em>
</p>


<p align="center">
  <img src="./assets/calib.gif" alt="图像描述" width="1200px"><br>
  <em>Fig. 3 Joint optimization of intrinsics and poses based on V-I data.</em>
</p>



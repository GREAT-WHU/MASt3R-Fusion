"""Measure the cam-IMU timeshift of an akai clip (per-clip, no calibration target).

Principle: the gyroscope and the camera observe the SAME rotation. The gyro gives
angular rate on the IMU clock; inter-frame optical flow gives it on the frame
(fsync) clock. The time lag that best aligns the two signals = the cam-IMU timeshift.

Usage:
  python3 measure_timeshift.py --imu imu.jsonl --video left_000.mjpeg
  python3 measure_timeshift.py --imu imu.jsonl --video left.mjpeg \
        --calib calibration.json --nframes 2400 --gyr-lsb-dps 32.8

Convention (matches akai calibration.json): t_imu = t_camera + timeshift.
A negative result means the image content is EARLIER than its fsync timestamp
(the frame was delivered late) -> bake TIMESHIFT_S=<tau> when building the EuRoC.
"""
import argparse, json, numpy as np, cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu", required=True, help="akai imu.jsonl")
    ap.add_argument("--video", required=True, help="left mjpeg/mp4 (cam0)")
    ap.add_argument("--calib", default=None, help="calibration.json (optional, prints stated value)")
    ap.add_argument("--nframes", type=int, default=2400, help="frames to analyse (~80s)")
    ap.add_argument("--gyr-lsb-dps", type=float, default=32.8, help="gyro scale LSB per deg/s")
    ap.add_argument("--max-lag", type=float, default=0.20, help="search range +/- s")
    ap.add_argument("--windows", type=int, default=4, help="reproducibility windows")
    args = ap.parse_args()

    t_us, gyro, ft = [], [], []
    for ln in open(args.imu):
        j = json.loads(ln)
        t_us.append(j["t_us"]); gyro.append((j["gx"], j["gy"], j["gz"]))
        if j.get("fsync_flag", 0) == 1:
            ft.append(j["t_us"] - j.get("fsync_delay_us", 0))
    t_us = np.array(t_us, float) / 1e6
    gmag = np.linalg.norm(np.array(gyro, float) / args.gyr_lsb_dps * np.pi / 180, axis=1)
    ft = np.array(ft, float) / 1e6

    cap = cv2.VideoCapture(args.video)
    prev = None; cam = []; ct = []; i = 0
    while i < args.nframes:
        ok, f = cap.read()
        if not ok: break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (480, 270))
        if prev is not None and i < len(ft):
            p0 = cv2.goodFeaturesToTrack(prev, 200, 0.01, 10)
            if p0 is not None:
                p1, stt, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None)
                good = stt.ravel() == 1
                if good.sum() > 10:
                    fl = np.linalg.norm((p1 - p0)[good].reshape(-1, 2), axis=1)
                    cam.append(np.median(fl)); ct.append(ft[i])
        prev = g; i += 1
    cap.release()
    cam = np.array(cam); ct = np.array(ct)
    camz = (cam - cam.mean()) / cam.std()

    lags = np.arange(-args.max_lag, args.max_lag + 1e-9, 0.001)
    def corr_at(tau, cz=camz, cts=ct):
        ga = np.interp(cts + tau, t_us, gmag); ga = (ga - ga.mean()) / ga.std()
        return np.corrcoef(cz, ga)[0, 1]
    cc = np.array([corr_at(x) for x in lags])
    tau = lags[cc.argmax()]; cpk = cc.max(); c0 = corr_at(0.0)

    print("frames analysed : %d" % len(cam))
    print("correlation @ tau=0        : %.3f" % c0)
    print("BEST lag tau               : %+.3f s   (correlation %.3f)" % (tau, cpk))
    W = len(ct) // args.windows
    taus = []
    for k in range(args.windows):
        sl = slice(k * W, (k + 1) * W)
        if sl.stop - sl.start < 30: continue
        ccw = np.array([corr_at(x, camz[sl], ct[sl]) for x in lags])
        taus.append(lags[ccw.argmax()])
        print("  window %d (%.0f-%.0fs): tau=%+.0f ms (corr %.2f)" %
              (k + 1, ct[sl][0], ct[sl][-1], lags[ccw.argmax()] * 1000, ccw.max()))
    if taus:
        print("reproducibility: tau = %+.0f +/- %.0f ms" % (np.mean(taus)*1000, np.std(taus)*1000))
    if args.calib:
        c = json.load(open(args.calib))
        print("calibration.json stated timeshift_cam_imu_s = %.4f s" %
              c.get("temporal", {}).get("timeshift_cam_imu_s", float("nan")))
    print("\n-> use TIMESHIFT_S=%.3f when building the EuRoC (t_imu = t_cam + tau)" % tau)

if __name__ == "__main__":
    main()
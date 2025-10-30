import dataclasses
import weakref
from pathlib import Path

import imgui
import lietorch
import torch
import moderngl
import moderngl_window as mglw
import numpy as np
from in3d.camera import Camera, ProjectionMatrix, lookat
from in3d.pose_utils import translation_matrix
from in3d.color import hex2rgba
from in3d.geometry import Axis
from in3d.viewport_window import ViewportWindow
from in3d.window import WindowEvents
from in3d.image import Image
from moderngl_window import resources
from moderngl_window.timers.clock import Timer

from mast3r_fusion.frame import Mode
from mast3r_fusion.geometry import get_pixel_coords
from mast3r_fusion.lietorch_utils import as_SE3
from mast3r_fusion.visualization_utils import (
    Frustums,
    Lines,
    depth2rgb,
    image_with_text,
)
from mast3r_fusion.config import load_config, config, set_global_config

from scipy.spatial.transform import Rotation
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm, expm

import numpy as np
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import cv2
def orthographic_matrix(w, h, scale, cx, cy, znear, zfar):
    left   = -w / (2 * scale) + (cx - w/2) / scale
    right  =  w / (2 * scale) + (cx - w/2) / scale
    bottom = -h / (2 * scale) + (cy - h/2) / scale
    top    =  h / (2 * scale) + (cy - h/2) / scale

    return np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / (right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)],
        [0, 0, -2.0 / (zfar - znear), -(zfar + znear) / (zfar - znear)],
        [0, 0, 0, 1]
    ], dtype=np.float32)


class OrthographicMatrix:
    def __init__(self, w, h, scale, cx, cy, znear, zfar):
        self.w = w
        self.h = h
        self.scale = scale
        self.cx = cx
        self.cy = cy
        self.znear = znear
        self.zfar = zfar
        self.cv2gl = np.array(
            [[1, 0, 0, 0],
             [0, -1, 0, 0],
             [0, 0, -1, 0],
             [0, 0, 0, 1]], dtype=np.float32
        )
        self.matrix = orthographic_matrix(w, h, scale, cx, cy, znear, zfar)

    def update(self, w=None, h=None, scale=None, cx=None, cy=None, znear=None, zfar=None):
        self.w = w or self.w
        self.h = h or self.h
        self.scale = scale or self.scale
        self.cx = cx or self.cx
        self.cy = cy or self.cy
        self.znear = znear or self.znear
        self.zfar = zfar or self.zfar
        self.matrix = orthographic_matrix(
            self.w, self.h, self.scale, self.cx, self.cy, self.znear, self.zfar
        )

    def gl_matrix(self):
        return self.matrix.T.copy()

def color_by_height_cmap(xyz, cmin=None, cmax=None, cmap_name='viridis'):
    z = xyz[:, 2]
    if cmin is None:
        cmin = np.min(z)
    if cmax is None:
        cmax = np.max(z)

    z_norm = np.clip((z - cmin) / (cmax - cmin), 0, 1)

    cmap = cm.get_cmap(cmap_name)
    colors = cmap(z_norm)[:, :3] 
    return colors

def average_pose_matrices(pose_matrices):
    if isinstance(pose_matrices, list):
        pose_matrices = np.stack(pose_matrices)

    translations = pose_matrices[:, :3, 3]
    mean_translation = np.mean(translations, axis=0)

    rotations = [R.from_matrix(pose[:3, :3]) for pose in pose_matrices]
    quaternions = np.stack([rot.as_quat() for rot in rotations])  # shape (K, 4)

    ref_q = quaternions[0]
    for i in range(1, len(quaternions)):
        if np.dot(quaternions[i], ref_q) < 0:
            quaternions[i] = -quaternions[i]

    mean_quat = np.mean(quaternions, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)
    mean_rot = R.from_quat(mean_quat).as_matrix()

    avg_pose = np.eye(4)
    avg_pose[:3, :3] = mean_rot
    avg_pose[:3, 3] = mean_translation

    return avg_pose


@dataclasses.dataclass
class WindowMsg:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5



def color_by_height(xyz, cmin=-2.0, cmax=2.0):
    z = np.clip((xyz[:,2] - cmin) / (cmax - cmin), 0, 1)
    colors = np.stack([z, np.zeros_like(z), 1.0 - z], axis=1)
    return colors



class Window(WindowEvents):
    title = "MASt3R-Fusion"
    window_size = (1960, 1080)

    def __init__(self, states, keyframes, main2viz, viz2main, lidar_points = None, img_name = None, max_show = 10, **kwargs):
        super().__init__(**kwargs)
        self.ctx.gc_mode = "auto"
        # bit hacky, but detect whether user is using 4k monitor
        self.scale = 1.0
        if self.wnd.buffer_size[0] > 2560:
            self.set_font_scale(2.0)
            self.scale = 2
        # self.clear = hex2rgba("#1E2326", alpha=1)
        self.clear = hex2rgba("#000000", alpha=1)
        # self.clear = hex2rgba("#FFFFFF", alpha=1)
        resources.register_dir((Path(__file__).parent.parent / "resources").resolve())

        self.line_prog = self.load_program("programs/lines.glsl")
        self.surfelmap_prog = self.load_program("programs/surfelmap.glsl")
        self.trianglemap_prog = self.load_program("programs/trianglemap.glsl")
        self.pointmap_prog = self.surfelmap_prog

        width, height = self.wnd.size
        self.camera = Camera(
            ProjectionMatrix(width, height, 75, width // 2, height // 2,1.0, 10000),
            lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0])),
        )
        # self.camera = Camera(
        #     OrthographicMatrix(width, height, scale=20.0, cx=width//2, cy=height//2, znear=1.0, zfar=100),
        #     lookat(np.array([2, 2, 2]), np.array([0, 0, 0]), np.array([0, 1, 0]))
        # )
        self.axis = Axis(self.line_prog, 0.1, 3 * self.scale)
        self.frustums = Frustums(self.line_prog)
        self.lines = Lines(self.line_prog)

        self.viewport = ViewportWindow("Scene", self.camera)
        self.state = WindowMsg()
        self.keyframes = keyframes
        self.states = states

        self.show_all = True
        self.show_keyframe_edges = False
        self.culling = True
        self.follow_cam = True

        self.depth_bias = 0.001
        self.frustum_scale = 0.05

        self.dP_dz = None

        self.line_thickness = 3
        self.show_keyframe = True
        self.show_curr_pointmap = True
        self.show_axis = True

        self.textures = dict()
        self.mtime = self.pointmap_prog.extra["meta"].resolved_path.stat().st_mtime
        self.curr_img, self.kf_img = Image(), Image()
        self.curr_img_np, self.kf_img_np = None, None

        self.main2viz = main2viz
        self.viz2main = viz2main
        self.count  = 0 
        self.cam_history =[]
        self.xyz = lidar_points
        self.max_show = max_show

        self.custom_prog = self.load_program(
            vertex_shader="programs/custom_pointcloud.glsl",
            fragment_shader="programs/custom_pointcloud.frag"
        )


    def render(self, t: float, frametime: float):
        self.viewport.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        if self.culling:
            self.ctx.enable(moderngl.CULL_FACE)
        self.ctx.clear(*self.clear)

        self.ctx.point_size = 2
        if self.show_axis:
            self.axis.render(self.camera)
        curr_frame = self.states.get_frame()
        h, w = curr_frame.img_shape.flatten()
        self.frustums.make_frustum(h, w)

        self.curr_img_np = curr_frame.uimg.numpy()
        self.curr_img.write(self.curr_img_np)


        cam_T_WC = as_SE3(curr_frame.T_WC).cpu()
        TCCp = np.eye(4,4)
        TCCp[0:3,0:3] = Rotation.from_euler('x', np.deg2rad(-5)).as_matrix()
        TCCp[0:3,3] = np.array([0,-4,-40])
        # TCCp = np.eye(4,4)
        # TCCp[2,3] = -2
        if self.follow_cam:
            T_WC = cam_T_WC.matrix().numpy().astype(
                dtype=np.float32
            ) @ TCCp.astype(np.float32)
            self.cam_history.append(T_WC[0])
            T_WC_temp = average_pose_matrices(self.cam_history[-5:]).astype(np.float32)[None]
            # T_WC[0,3:3,3:3] = T_WC_temp[0,0:3,0:3]
            self.camera.follow_cam(np.linalg.inv(T_WC_temp))
        else:
            self.camera.unfollow_cam()
        self.frustums.add(
            cam_T_WC,
            scale=self.frustum_scale,
            color=[0, 1, 0, 1],
            thickness=self.line_thickness * self.scale,
        )


        with self.keyframes.lock:
            N_keyframes = self.keyframes.n_size.value + self.keyframes.rollup_sum.value
            dirty_idx = self.keyframes.get_dirty_idx()
        for kf_idx in dirty_idx:
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()
            X = self.frame_X(keyframe)
            C = keyframe.get_average_conf().cpu().numpy().astype(np.float32)

            if keyframe.frame_id not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures[keyframe.frame_id] = ptex, ctex, itex
                ptex, ctex, itex = self.textures[keyframe.frame_id]

                brightened_img = np.power(keyframe.uimg.numpy(), 0.9)
                brightened_img = np.clip(brightened_img, 0, 1)

                itex.write(brightened_img.astype(np.float32).tobytes())

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())

        for kf_idx in range(max(0,N_keyframes-self.max_show),N_keyframes):
            keyframe = self.keyframes[kf_idx]
            h, w = keyframe.img_shape.flatten()

            if kf_idx == N_keyframes - 1:
                self.kf_img_np = keyframe.uimg.numpy()
                self.kf_img.write(self.kf_img_np)

            color = [1, 0, 0, 1]
            if self.show_keyframe:
                self.frustums.add(
                    as_SE3(keyframe.T_WC.cpu()),
                    scale=self.frustum_scale,
                    color=color,
                    thickness=self.line_thickness * self.scale,
                )

            ptex, ctex, itex = self.textures[keyframe.frame_id]
            if self.show_all:

                if not (self.xyz is None):
                    colors = color_by_height_cmap(self.xyz, cmin=-2.0, cmax=1.0, cmap_name='jet')
                    self.render_custom_pointcloud(self.xyz, colors=colors, ref_keyframe_idx=0, point_size=3.0)
                else:
                    self.render_pointmap(keyframe.T_WC.cpu(), w, h, ptex, ctex, itex)



                # self.render_custom_pointcloud(points, colors, ref_keyframe_idx=5, point_size=3.0)
                # colors = color_by_height_cmap(self.xyz, cmin=-2.0, cmax=2.0, cmap_name='jet')
                # self.render_custom_pointcloud(self.xyz, colors=colors, ref_keyframe_idx=0, point_size=3.0)



        if self.show_keyframe_edges:
            with self.states.lock:
                ii = torch.tensor(self.states.edges_ii, dtype=torch.long)
                jj = torch.tensor(self.states.edges_jj, dtype=torch.long)
                if ii.numel() > 0 and jj.numel() > 0:
                    T_WCi = lietorch.Sim3(self.keyframes.T_WC[ii, 0])
                    T_WCj = lietorch.Sim3(self.keyframes.T_WC[jj, 0])
            if ii.numel() > 0 and jj.numel() > 0:
                t_WCi = T_WCi.matrix()[:, :3, 3].cpu().numpy()
                t_WCj = T_WCj.matrix()[:, :3, 3].cpu().numpy()
                self.lines.add(
                    t_WCi,
                    t_WCj,
                    thickness=self.line_thickness * self.scale,
                    color=[0, 1, 0, 1],
                )
        if self.show_curr_pointmap and self.states.get_mode() != Mode.INIT:
            if config["use_calib"]:
                curr_frame.K = self.keyframes.get_intrinsics()
            h, w = curr_frame.img_shape.flatten()
            X = self.frame_X(curr_frame)
            C = curr_frame.C.cpu().numpy().astype(np.float32)

            if "curr" not in self.textures:
                ptex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                ctex = self.ctx.texture((w, h), 1, dtype="f4", alignment=4)
                itex = self.ctx.texture((w, h), 3, dtype="f4", alignment=4)
                self.textures["curr"] = ptex, ctex, itex
            ptex, ctex, itex = self.textures["curr"]
            ptex.write(X.tobytes())
            ctex.write(C.tobytes())
            itex.write(depth2rgb(X[..., -1], colormap="turbo"))
            self.render_pointmap(
                curr_frame.T_WC.cpu(),
                w,
                h,
                ptex,
                ctex,
                itex,
                use_img=True,
                depth_bias=self.depth_bias,
            )

        self.lines.render(self.camera)
        self.frustums.render(self.camera)
        self.render_ui()
        # img_data = self.ctx.screen.read(components=3, alignment=1)
        # image = np.frombuffer(img_data, dtype=np.uint8).reshape(self.wnd.buffer_size[1], self.wnd.buffer_size[0], 3)[::-1]
        # from PIL import Image as PILImage
        # PILImage.fromarray(image).save("video/%010d.jpg"%self.count)
        # self.count += 1
        # print("Screenshot saved to screenshot.png")

    def render_ui(self):
        self.wnd.use()
        imgui.new_frame()

        io = imgui.get_io()
        # get window size and full screen
        window_size = io.display_size
        imgui.set_next_window_size(window_size[0], window_size[1])
        imgui.set_next_window_position(0, 0)
        self.viewport.render()

        imgui.set_next_window_size(
            window_size[0] / 4, 15 * window_size[1] / 16, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_position(
            32 * self.scale, 32 * self.scale, imgui.FIRST_USE_EVER
        )
        imgui.set_next_window_focus()
        imgui.begin("GUI", flags=imgui.WINDOW_ALWAYS_VERTICAL_SCROLLBAR)
        new_state = WindowMsg()
        _, new_state.is_paused = imgui.checkbox("pause", self.state.is_paused)

        imgui.spacing()
        _, new_state.C_conf_threshold = imgui.slider_float(
            "C_conf_threshold", self.state.C_conf_threshold, 0, 10
        )

        imgui.spacing()

        _, self.show_all = imgui.checkbox("show all", self.show_all)
        imgui.same_line()
        _, self.follow_cam = imgui.checkbox("follow cam", self.follow_cam)

        imgui.spacing()
        shader_options = [
            "surfelmap.glsl",
            "trianglemap.glsl",
            "lidar_pointcloud.glsl",
        ]
        current_shader = shader_options.index(
            self.pointmap_prog.extra["meta"].resolved_path.name
        )

        for i, shader in enumerate(shader_options):
            if imgui.radio_button(shader, current_shader == i):
                current_shader = i

        selected_shader = shader_options[current_shader]
        if selected_shader != self.pointmap_prog.extra["meta"].resolved_path.name:
            self.pointmap_prog = self.load_program(f"programs/{selected_shader}")

        imgui.spacing()

        _, self.show_keyframe_edges = imgui.checkbox(
            "show_keyframe_edges", self.show_keyframe_edges
        )
        imgui.spacing()

        _, self.pointmap_prog["show_normal"].value = imgui.checkbox(
            "show_normal", self.pointmap_prog["show_normal"].value
        )
        imgui.same_line()
        _, self.culling = imgui.checkbox("culling", self.culling)
        if "radius" in self.pointmap_prog:
            _, self.pointmap_prog["radius"].value = imgui.drag_float(
                "radius",
                self.pointmap_prog["radius"].value,
                0.0001,
                min_value=0.0,
                max_value=0.1,
            )
        if "slant_threshold" in self.pointmap_prog:
            _, self.pointmap_prog["slant_threshold"].value = imgui.drag_float(
                "slant_threshold",
                self.pointmap_prog["slant_threshold"].value,
                0.1,
                min_value=0.0,
                max_value=1.0,
            )
        _, self.show_keyframe = imgui.checkbox("show_keyframe", self.show_keyframe)
        _, self.show_curr_pointmap = imgui.checkbox(
            "show_curr_pointmap", self.show_curr_pointmap
        )
        _, self.show_axis = imgui.checkbox("show_axis", self.show_axis)
        _, self.line_thickness = imgui.drag_float(
            "line_thickness", self.line_thickness, 0.1, 10, 0.5
        )

        _, self.frustum_scale = imgui.drag_float(
            "frustum_scale", self.frustum_scale, 0.001, 0, 0.1
        )

        imgui.spacing()

        gui_size = imgui.get_content_region_available()
        scale = gui_size[0] / self.curr_img.texture.size[0]
        scale = min(self.scale, scale)
        size = (
            self.curr_img.texture.size[0] * scale,
            self.curr_img.texture.size[1] * scale,
        )
        image_with_text(self.kf_img, size, "kf", same_line=False)
        image_with_text(self.curr_img, size, "curr", same_line=False)

        imgui.end()

        if new_state != self.state:
            self.state = new_state
            self.send_msg()

        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def send_msg(self):
        self.viz2main.put(self.state)

    def render_custom_pointcloud(
        self,
        points: np.ndarray,
        colors: np.ndarray = None,
        ref_keyframe_idx: int = None,
        point_size: float = 2.0,
    ):
        assert points.ndim == 2 and points.shape[1] == 3, "points 必须是 (N,3)"
        N = points.shape[0]

        if colors is None:
            colors = np.ones_like(points, dtype=np.float32)
        else:
            assert colors.shape == points.shape, "colors 必须和 points 形状一致"

        if ref_keyframe_idx is not None:
            keyframe = self.keyframes[ref_keyframe_idx]
            T_WC = as_SE3(keyframe.T_WC.cpu()).matrix().numpy().astype(np.float32)
        else:
            T_WC = np.eye(4, dtype=np.float32)
        print(T_WC)
        T_LC = np.array([[0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
                         [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
                         [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
                         [0,0,0,1]])
        
        T_WC = T_WC @ np.linalg.inv(T_LC)

        vbo_points = self.ctx.buffer(points.astype(np.float32).tobytes())
        vbo_colors = self.ctx.buffer(colors.astype(np.float32).tobytes())

        vao_content = [
            (vbo_points, "3f", "in_position"),
            (vbo_colors, "3f", "in_color"),
        ]
        vao = self.ctx.vertex_array(self.custom_prog, vao_content)

        T_WC_mat = np.ascontiguousarray(T_WC.T, dtype=np.float32)
        vao.program["m_model"].write(T_WC_mat)
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())
        vao.program["point_size"].value = point_size

        self.ctx.point_size = point_size
        vao.render(mode=moderngl.POINTS, vertices=N)

        vao.release()
        vbo_points.release()
        vbo_colors.release()

    def render_pointmap(self, T_WC, w, h, ptex, ctex, itex, use_img=True, depth_bias=0):
        w, h = int(w), int(h)
        ptex.use(0)
        ctex.use(1)
        itex.use(2)
        model = T_WC.matrix().numpy().astype(np.float32).T

        vao = self.ctx.vertex_array(self.pointmap_prog, [], skip_errors=True)
        vao.program["m_camera"].write(self.camera.gl_matrix())
        vao.program["m_model"].write(model)
        vao.program["m_proj"].write(self.camera.proj_mat.gl_matrix())

        vao.program["pointmap"].value = 0
        vao.program["confs"].value = 1
        vao.program["img"].value = 2
        vao.program["width"].value = w
        vao.program["height"].value = h
        vao.program["conf_threshold"] = self.state.C_conf_threshold
        vao.program["use_img"] = use_img
        if "depth_bias" in self.pointmap_prog:
            vao.program["depth_bias"] = depth_bias
        vao.render(mode=moderngl.POINTS, vertices=w * h)
        vao.release()

    def frame_X(self, frame):
        if config["use_calib"]:
            Xs = frame.X_canon[None]
            if self.dP_dz is None:
                device = Xs.device
                dtype = Xs.dtype
                img_size = frame.img_shape.flatten()[:2]
                K = frame.K
                p = get_pixel_coords(
                    Xs.shape[0], img_size, device=device, dtype=dtype
                ).view(*Xs.shape[:-1], 2)
                tmp1 = (p[..., 0] - K[0, 2]) / K[0, 0]
                tmp2 = (p[..., 1] - K[1, 2]) / K[1, 1]
                self.dP_dz = torch.empty(
                    p.shape[:-1] + (3, 1), device=device, dtype=dtype
                )
                self.dP_dz[..., 0, 0] = tmp1
                self.dP_dz[..., 1, 0] = tmp2
                self.dP_dz[..., 2, 0] = 1.0
                self.dP_dz = self.dP_dz[..., 0].cpu().numpy().astype(np.float32)
            return (Xs[..., 2:3].cpu().numpy().astype(np.float32) * self.dP_dz)[0]

        return frame.X_canon.cpu().numpy().astype(np.float32)


def run_visualization(cfg, states, keyframes, main2viz, viz2main, lidar_points = None, img_name = None, max_show = 10) -> None:
    set_global_config(cfg)

    config_cls = Window
    backend = "glfw"
    window_cls = mglw.get_local_window_cls(backend)

    window = window_cls(
        title=config_cls.title,
        size=config_cls.window_size,
        fullscreen=False,
        resizable=True,
        visible=True,
        gl_version=(3, 3),
        aspect_ratio=None,
        vsync=True,
        samples=4,
        cursor=True,
        backend=backend
    )
    window.print_context_info()
    mglw.activate_context(window=window)
    window.ctx.gc_mode = "auto"
    timer = Timer()
    window_config = config_cls(
        states=states,
        keyframes=keyframes,
        main2viz=main2viz,
        viz2main=viz2main,
        ctx=window.ctx,
        wnd=window,
        timer=timer,
        lidar_points = lidar_points,
        img_name = img_name,
        max_show = max_show
    )
    # Avoid the event assigning in the property setter for now
    # We want the even assigning to happen in WindowConfig.__init__
    # so users are free to assign them in their own __init__.
    window._config = weakref.ref(window_config)

    # Swap buffers once before staring the main loop.
    # This can trigged additional resize events reporting
    # a more accurate buffer size
    window.swap_buffers()
    window.set_default_viewport()

    timer.start()

    while not window.is_closing:
        if states.get_mode() == Mode.TERMINATED:
            break
        current_time, delta = timer.next_frame()

        if window_config.clear_color is not None:
            window.clear(*window_config.clear_color)

        # Always bind the window framebuffer before calling render
        window.use()

        window.render(current_time, delta)
        if not window.is_closing:
            window.swap_buffers()

    state = window_config.state
    window.destroy()
    state.is_terminated = True
    viz2main.put(state)

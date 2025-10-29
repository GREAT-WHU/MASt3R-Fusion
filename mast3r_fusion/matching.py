import torch
import torch.nn.functional as F
import mast3r_fusion.image as img_utils
from mast3r_fusion.config import config
import mast3r_fusion_backends
import time


def match(X11, X21, D11, D21, idx_1_to_2_init=None, subpixel_factor = 1):
    idx_1_to_2, valid_match2 = match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init, subpixel_factor)
    return idx_1_to_2, valid_match2


def pixel_to_lin(p1, w, subpixel_factor = 1):
    idx_1_to_2 = p1[..., 0] + (w * subpixel_factor * p1[..., 1])
    return idx_1_to_2


def lin_to_pixel(idx_1_to_2, w):
    u = idx_1_to_2 % w
    v = idx_1_to_2 // w
    p = torch.stack((u, v), dim=-1)
    return p


def prep_for_iter_proj(X11, X21, idx_1_to_2_init):
    b, h, w, _ = X11.shape
    device = X11.device

    # Ray image
    rays_img = F.normalize(X11, dim=-1)
    rays_img = rays_img.permute(0, 3, 1, 2)  # (b,c,h,w)
    gx_img, gy_img = img_utils.img_gradient(rays_img)
    rays_with_grad_img = torch.cat((rays_img, gx_img, gy_img), dim=1)
    rays_with_grad_img = rays_with_grad_img.permute(
        0, 2, 3, 1
    ).contiguous()  # (b,h,w,c)

    # 3D points to project
    X21_vec = X21.view(b, -1, 3)
    pts3d_norm = F.normalize(X21_vec, dim=-1)

    # Initial guesses of projections
    if idx_1_to_2_init is None:
        # Reset to identity mapping
        idx_1_to_2_init = torch.arange(h * w, device=device)[None, :].repeat(b, 1)
    p_init = lin_to_pixel(idx_1_to_2_init, w)
    p_init = p_init.float()

    return rays_with_grad_img, pts3d_norm, p_init


def match_iterative_proj(X11, X21, D11, D21, idx_1_to_2_init=None, subpixel_factor = 1):
    cfg = config["matching"]
    b, h, w = X21.shape[:3]
    device = X11.device
    print('--',time.time())
    rays_with_grad_img, pts3d_norm, p_init = prep_for_iter_proj(
        X11, X21, idx_1_to_2_init
    )
    p1, valid_proj2 = mast3r_fusion_backends.iter_proj(
        rays_with_grad_img,
        pts3d_norm,
        p_init,
        cfg["max_iter"],
        cfg["lambda_init"],
        cfg["convergence_thresh"],
    )
    p1 = p1.long()

    # Check for occlusion based on distances
    batch_inds = torch.arange(b, device=device)[:, None].repeat(1, h * w)

    # X_temp1 = X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3).clone()
    # X_temp1 = X_temp1[:,:,:,:2]/X_temp1[:,:,:,2:3] * 200
    # X_temp2 = X21.clone()
    # X_temp2 = X_temp2[:,:,:,:2]/X_temp2[:,:,:,2:3] * 200
    dists2 = torch.linalg.norm(
        X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3) - X21, dim=-1
    )
    # dists2 = torch.linalg.norm(
    #     X_temp1 - X_temp2, dim=-1
    # )
    valid_dists2 = (dists2 < cfg["dist_thresh"]).view(b, -1)
    # valid_dists2 = (dists2 < 0.4).view(b, -1)
    # valid_far = (X11[batch_inds, p1[..., 1], p1[..., 0], :].reshape(b, h, w, 3)[:,:,:,2] > 50.0).view(b, -1)
    # valid_dists2 = torch.logical_or(valid_dists2,valid_far)
    valid_proj2 = valid_proj2 & valid_dists2
    print(time.time())

    if cfg["radius"] > 0:
        (p1,) = mast3r_fusion_backends.refine_matches(
            D11.half(),
            D21.view(b, h * w, -1).half(),
            p1,
            cfg["radius"],
            cfg["dilation_max"],
        )
    print(time.time())
    

    
    if X11.shape[0] > 1:
        123
    # print(time.time())


    if subpixel_factor == 4:
        D11_up = F.interpolate(
            D11.permute(0, 3, 1, 2),  # 转成 [B, C, H, W]
            scale_factor=4,
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1)  # 再转回 [B, H, W, C]

        for ibatch in range(p1.shape[0]):
            N, H, W, C = D21.shape  # N=1
            _, H2, W2, _ = D11_up.shape  # H2 = H*2, W2 = W*2

            # p1: [1, H*W, 2]，存x,y坐标
            p1_coords = p1[ibatch].long()  # (H*W, 2)

            # 放大坐标2倍，转成y,x方便索引，注意原来p1是[x,y]还是[y,x]
            # 这里按代码理解p1[:, :, 0]是x，p1[:, :, 1]是y
            base_x = p1_coords[:, 0] * subpixel_factor  # (H*W,)
            base_y = p1_coords[:, 1] * subpixel_factor  # (H*W,)

            # 创建3x3偏移
            offsets = torch.tensor([[-2,-2],[-2,-1], [-2,0], [-2,1],[-2,2],
                                    [-1,-2],[-1,-1], [-1,0], [-1,1],[-1,2],
                                    [-0,-2],[ 0,-1],  [0,0],  [0,1], [0,2], 
                                    [ 1,-2],[ 1,-1],  [1,0],  [1,1], [1,2], 
                                    [ 2,-2],[ 2,-1],  [2,0],  [2,1], [2,2]  ], device=p1.device)  # (9, 2)

            # 对base坐标加偏移，得到采样点
            # (H*W, 1, 2) + (1, 9, 2) => (H*W, 9, 2)
            sample_coords = torch.stack([base_y, base_x], dim=1).unsqueeze(1) + offsets.unsqueeze(0)

            # 边界裁剪
            sample_coords[..., 0] = sample_coords[..., 0].clamp(0, H2 - 1)
            sample_coords[..., 1] = sample_coords[..., 1].clamp(0, W2 - 1)

            # 将采样坐标拆成两个索引张量
            sample_y = sample_coords[..., 0].long()  # (H*W, 9)
            sample_x = sample_coords[..., 1].long()  # (H*W, 9)

            # D21展开成(H*W, C)
            D21_flat = D21[ibatch].view(H*W, C)  # (H*W, C)

            # 从D11_up采样对应点特征 (H*W, 9, C)
            # 通过高级索引：
            D11_up_flat = D11_up[ibatch]  # (H2, W2, C)
            # 使用 gather 之前转成 (H2*W2, C)
            D11_up_reshape = D11_up_flat.view(H2*W2, C)  # (H2*W2, C)
            indices = sample_y * W2 + sample_x  # (H*W, 9)
            print(time.time())

            # 索引
            D11_samples = D11_up_reshape[indices]  # (H*W, 9, C)

            # 计算点积： (H*W, 9)
            dot_products = (D11_samples * D21_flat.unsqueeze(1)).sum(dim=2)

            # 找最大点积的索引
            max_dot_vals, max_idx = dot_products.max(dim=1)  # (H*W,)

            # 选出对应的偏移
            best_offsets = offsets[max_idx]  # (H*W, 2)

            # 计算最终坐标
            final_coords = torch.stack([base_x, base_y], dim=1) + best_offsets  # (H*W, 2)

            # 保存到p1_sub
            p1[ibatch, :, 0] = torch.clamp(final_coords[:, 0], min=0)
            p1[ibatch, :, 1] = torch.clamp(final_coords[:, 1], min=0)
            print(time.time())
    elif subpixel_factor == 2:
        D11_up = F.interpolate(
            D11.permute(0, 3, 1, 2),  # 转成 [B, C, H, W]
            scale_factor=2,
            mode='bilinear',
            align_corners=True
        ).permute(0, 2, 3, 1)  # 再转回 [B, H, W, C]

        for ibatch in range(p1.shape[0]):
            N, H, W, C = D21.shape  # N=1
            _, H2, W2, _ = D11_up.shape  # H2 = H*2, W2 = W*2

            # p1: [1, H*W, 2]，存x,y坐标
            p1_coords = p1[ibatch].long()  # (H*W, 2)

            # 放大坐标2倍，转成y,x方便索引，注意原来p1是[x,y]还是[y,x]
            # 这里按代码理解p1[:, :, 0]是x，p1[:, :, 1]是y
            base_x = p1_coords[:, 0] * 2  # (H*W,)
            base_y = p1_coords[:, 1] * 2  # (H*W,)

            # 创建3x3偏移
            offsets = torch.tensor([[-1,-1], [-1,0], [-1,1],
                                    [0,-1],  [0,0],  [0,1],
                                    [1,-1],  [1,0],  [1,1]], device=p1.device)  # (9, 2)

            # 对base坐标加偏移，得到采样点
            # (H*W, 1, 2) + (1, 9, 2) => (H*W, 9, 2)
            sample_coords = torch.stack([base_y, base_x], dim=1).unsqueeze(1) + offsets.unsqueeze(0)

            # 边界裁剪
            sample_coords[..., 0] = sample_coords[..., 0].clamp(0, H2 - 1)
            sample_coords[..., 1] = sample_coords[..., 1].clamp(0, W2 - 1)

            # 将采样坐标拆成两个索引张量
            sample_y = sample_coords[..., 0].long()  # (H*W, 9)
            sample_x = sample_coords[..., 1].long()  # (H*W, 9)

            # D21展开成(H*W, C)
            D21_flat = D21[ibatch].view(H*W, C)  # (H*W, C)

            # 从D11_up采样对应点特征 (H*W, 9, C)
            # 通过高级索引：
            D11_up_flat = D11_up[ibatch]  # (H2, W2, C)
            # 使用 gather 之前转成 (H2*W2, C)
            D11_up_reshape = D11_up_flat.view(H2*W2, C)  # (H2*W2, C)
            indices = sample_y * W2 + sample_x  # (H*W, 9)
            print(time.time())

            # 索引
            D11_samples = D11_up_reshape[indices]  # (H*W, 9, C)

            # 计算点积： (H*W, 9)
            dot_products = (D11_samples * D21_flat.unsqueeze(1)).sum(dim=2)

            # 找最大点积的索引
            max_dot_vals, max_idx = dot_products.max(dim=1)  # (H*W,)

            # 选出对应的偏移
            best_offsets = offsets[max_idx]  # (H*W, 2)

            # 计算最终坐标
            final_coords = torch.stack([base_x, base_y], dim=1) + best_offsets  # (H*W, 2)

            # 保存到p1_sub
            p1[ibatch, :, 0] = torch.clamp(final_coords[:, 0], min=0)
            p1[ibatch, :, 1] = torch.clamp(final_coords[:, 1], min=0)
            print(time.time())
    elif subpixel_factor == 1:
        pass
    else:
        raise Exception("subpixel_factor must be 1 or 2")

    # Convert to linear index
    idx_1_to_2 = pixel_to_lin(p1, w, subpixel_factor)
    return idx_1_to_2, valid_proj2.unsqueeze(-1)

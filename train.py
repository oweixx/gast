import torch.nn.functional as F
import cv2
import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint
import copy

import torch
from tqdm import tqdm
import torchvision
from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import torchvision.models as models
import numpy as np
from torchvision.models import VGG16_Weights
from torchvision.utils import save_image
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from utils.color_matching import color_match, init_scene_images
from utils.nnfm_loss_contra import NNFMLossContra
import warnings
warnings.filterwarnings('ignore')

VGG = models.vgg16(weights=VGG16_Weights.DEFAULT).eval().to("cuda")
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to("cuda").eval()

def training(dataset, opt, pipe, args):
    torch.cuda.empty_cache()
    testing_iterations = args.test_iterations
    checkpoint_iterations = args.checkpoint_iterations
    checkpoint = args.checkpoint
    debug_from = args.debug_from
    style_image = args.style_image
    style_image_path = args.style_image
    
    # Load style image
    sample_size = args.sample_size
    style_image = cv2.imread(style_image_path)
    image_rgb = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    scene_height, scene_width = image_rgb.shape[:2]
    scale_ratio = sample_size / scene_height
    target_width = int(scene_width * scale_ratio)
    
    resized_image = cv2.resize(image_rgb, (target_width, sample_size), interpolation=cv2.INTER_AREA)
    resized_image = torch.from_numpy(resized_image).permute(2, 0, 1).float() / 255.0
    resized_image = resized_image.to("cuda")
    
    with torch.no_grad() :
        style_image = cv2.imread(style_image_path)
        style_depth_hw = model.infer_image(style_image)
        style_depth = torch.from_numpy(style_depth_hw).float()
        
        style_depth = style_depth.unsqueeze(0).unsqueeze(0)
        style_depth = F.interpolate(style_depth, size=(sample_size, target_width), mode='bilinear', align_corners=False)

        '''min-max normalize'''
        d_min = style_depth.amin(dim=(2,3), keepdim=True)
        d_max = style_depth.amax(dim=(2,3), keepdim=True)
        style_depth = (style_depth - d_min) / (d_max - d_min + 1e-8)
        
        style_depth = 1.0 - style_depth
        style_depth = style_depth.squeeze(0)
        style_depth = torch.cat([style_depth]*3, dim = 0)
        style_depth = style_depth.to("cuda")
        #save_image(style_depth, "style_depth.png")
        
        style_depth_np_og = style_depth.cpu().numpy()
        style_depth_np = style_depth_np_og[0]
        minv = style_depth_np.min()
        maxv = style_depth_np.max()
        style_depth_np = (style_depth_np - minv) / (maxv - minv + 1e-8)
        img_out = (style_depth_np * 255).astype('uint8')
        img_out = cv2.applyColorMap(img_out, cv2.COLORMAP_JET)
        #cv2.imwrite("style_depth_fixed.png", img_out)
    
    '''style edge extract'''
    canny_low = 25 ; canny_high = 200
    style_edge = []
    
    edge = cv2.Canny(img_out, canny_low, canny_high)
    style_edge.append(edge)
    style_edge = np.stack(style_edge, axis=-1)
    style_edge = np.mean(style_edge, axis=-1)
    
    style_edge = torch.from_numpy(style_edge).unsqueeze(0).unsqueeze(0).float() / 255.0
    style_edge = F.interpolate(style_edge, size=(sample_size, target_width), mode='bilinear', align_corners=False)
    style_edge = style_edge.squeeze(0)
    style_edge = torch.cat([style_edge]*3, dim = 0)
    style_edge = style_edge.to("cuda")
    #save_image(style_edge, "style_edge.png")
    
    style_img_np = cv2.imread(style_image_path, cv2.IMREAD_COLOR)
    style_img_np = cv2.cvtColor(style_img_np, cv2.COLOR_BGR2RGB)
    style_img_np = style_img_np.astype(np.float32) / 255.0
    style_image = torch.from_numpy(style_img_np).float()

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.load_ply(args.ply_path)
    gaussians.training_setup(opt)
    
    depth_dir = os.path.join(args.model_path, "depth")
    os.makedirs(depth_dir, exist_ok = True)
                  
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    if not args.geometry_stylization :
        ema_loss_for_log = 0.0
        progress_bar = tqdm(range(first_iter, opt.matching_iterations),
                            desc="Color Matching Progress")
        
        scene_images = init_scene_images(scene)
        scene_images = scene_images.permute(0, 2, 3, 1)
        scene_images = scene_images.contiguous()
        image_set, _ = color_match(scene_images, style_image)
        
        viewpoint_stack = scene.getTrainCameras()
        for i, view in enumerate(viewpoint_stack):
            image = image_set[i]
            image = image.permute(2,0,1).contiguous().float()
            view.original_image = image.clone()
        
        viewpoint_stack = None
        first_iter += 1
        
        '''Color Match'''
        for iteration in range(first_iter, opt.matching_iterations + 1):
            if network_gui.conn == None:
                network_gui.try_connect()
            while network_gui.conn != None:
                try:
                    net_image_bytes = None
                    custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                    if custom_cam != None:
                        net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)[
                            "render"]
                        net_image_bytes = memoryview((torch.clamp(
                            net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    network_gui.send(net_image_bytes, dataset.source_path)
                    if do_training and ((iteration < int(opt.matching_iterations)) or not keep_alive):
                        break
                except Exception as e:
                    network_gui.conn = None

            iter_start.record()
            gaussians.update_learning_rate(iteration)

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

            if (iteration - 1) == debug_from:
                pipe.debug = True
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            image = render_pkg["render"]
            gt_image = viewpoint_cam.original_image.cuda()    
            
            Ll1 = l1_loss(image, gt_image)
            ssim_value = ssim(image, gt_image)
            
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
            
            loss.backward()
            iter_end.record()
            
            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(10)
                if iteration == opt.matching_iterations:
                    progress_bar.close()

                training_report(tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(
                    iter_end), testing_iterations, scene, render, (pipe, background))

                if iteration < opt.matching_iterations:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)
    
    gt_gaussians = copy.deepcopy(gaussians)
    for param in gt_gaussians.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.requires_grad_(False)
            
    with torch.no_grad():
        for cam in scene.getTrainCameras().copy():
            gt_image = cam.original_image.cuda()
            cam.vgg_features = get_features(gt_image)
    
    params_col = [
        {'params': [gaussians._features_dc], 'lr': opt.feature_lr, "name": "f_dc"}
    ]
    optimizer_col = torch.optim.Adam(params_col, lr=0.0, eps=1e-15)
    
    params_geo = [
        {'params': [gaussians._xyz], 'lr': opt.position_lr_init  * scene.cameras_extent, "name": "xyz"},
        {'params': [gaussians._scaling], 'lr': opt.scaling_lr, "name": "scaling"},
        {'params': [gaussians._rotation], 'lr': opt.rotation_lr, "name": "rotation"},
        {'params': [gaussians._opacity], 'lr': opt.opacity_lr, "name": "opacity"}
    ]
    optimizer_geo = torch.optim.Adam(params_geo, lr=0.0, eps=1e-15)
    
    nnfm_loss_contra_fn = NNFMLossContra("cuda")
    
    first_iter = 0
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations),
                    desc="Decoupled Optimized Progress")
    first_iter += 1
    
    viewpoint_stack = []
    if not viewpoint_stack:
        viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
    
    inner_iterations = 100
    outer_iterations = opt.iterations // inner_iterations
    
    if not args.geometry_stylization : 
        N_g = int(inner_iterations * opt.ratio_geo)
        N_c = inner_iterations - N_g
    else :
        N_g = 100
        N_c = 0
    iteration = 0
    
    '''Geometry Update'''
    for _ in range(outer_iterations):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)[
                        "render"]
                    net_image_bytes = memoryview((torch.clamp(
                        net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
                
        iter_start.record()
        gaussians.update_learning_rate(iteration)
        bg = torch.rand((3), device="cuda") if opt.random_background else background
        
        # fixed view for debugging
        if iteration == 0 :
            fix_view = viewpoint_cam

        '''debugging'''
        if args.debug_stylized :
            fix_view_render_pkg = render(fix_view, gaussians, pipe, bg)
            depth = fix_view_render_pkg["depth"]
            image_np = depth.squeeze(0).detach().cpu().numpy()
            image_normalized = (image_np - image_np.min()) / (image_np.max() - image_np.min())
            image_uint8 = (image_normalized * 255).astype(np.uint8)
            image_colored = cv2.applyColorMap(image_uint8, cv2.COLORMAP_JET)
            cv2.imwrite(f"{depth_dir}/depth_{iteration}.png", image_colored)
            
            image = fix_view_render_pkg["render"]
            save_image(image, f"{depth_dir}/render_{iteration}.png")
        
        for _ in range(N_c) :
            iteration += 1
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            with torch.no_grad() :
                gt_pkg = render(viewpoint_cam, gt_gaussians, pipe, bg)
            
            base_loss = compute_color_base(render_pkg, viewpoint_cam, opt)
            geo_loss = compute_geometry_loss(render_pkg, nnfm_loss_contra_fn, resized_image, style_depth, style_edge, opt)
            
            loss = geo_loss + base_loss
            loss.backward()
            optimizer_col.step()
            optimizer_col.zero_grad(set_to_none=True)

            if iteration % 10 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)

        for _ in range(N_g) :
            iteration += 1
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            
            with torch.no_grad() :
                gt_pkg = render(viewpoint_cam, gt_gaussians, pipe, bg)
            
            render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
            base_loss = compute_geo_base(render_pkg, gt_pkg, viewpoint_cam, gaussians, gt_gaussians, opt)
            geo_loss = compute_geometry_loss(render_pkg, nnfm_loss_contra_fn, resized_image, style_depth, style_edge, opt)
            
            loss = base_loss + geo_loss
            
            loss.backward()
            optimizer_geo.step()
            optimizer_geo.zero_grad(set_to_none=True)
            
            if iteration % 10 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            
        iter_end.record()

        with torch.no_grad():
            if iteration == opt.iterations:
                progress_bar.close()

            training_report(tb_writer, iteration, loss, l1_loss, iter_start.elapsed_time(
                iter_end), testing_iterations, scene, render, (pipe, background))
            
            if (iteration == opt.iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration),
                        scene.model_path + "/chkpnt" + str(iteration) + ".pth")      

def compute_color_base(render_pkg, viewpoint_cam, opt) :
    image = render_pkg["render"]
    
    '''Content Loss'''
    gt_features = viewpoint_cam.vgg_features
    render_features =  get_features(image)
    content_loss = torch.mean((render_features - gt_features)**2)
    
    '''tv reg'''
    tmp_image = image.unsqueeze(0)
    w_variance = torch.mean(torch.pow(tmp_image[:, :, :-1] - tmp_image[:, :, 1:], 2))
    h_variance = torch.mean(torch.pow(tmp_image[:, :-1, :] - tmp_image[:, 1:, :], 2))
    img_tv_loss = (h_variance + w_variance) / 2.0
    
    return  content_loss * opt.lambda_con + img_tv_loss * 0.02
            
def compute_geo_base(render_pkg, gt_pkg, viewpoint_cam, gaussians, gt_gaussians, opt) :
    image = render_pkg["render"]
    
    '''content loss'''
    gt_features = viewpoint_cam.vgg_features
    render_features =  get_features(image)
    content_loss = torch.mean((render_features - gt_features)**2)
    
    '''depth reg'''
    gt_depth = gt_pkg["depth"]
    depth = render_pkg["depth"]
    global_depth_reg = l1_loss(depth, gt_depth) 
    
    '''tv reg'''
    tmp_image = image.unsqueeze(0)
    w_variance = torch.mean(torch.pow(tmp_image[:, :, :-1] - tmp_image[:, :, 1:], 2))
    h_variance = torch.mean(torch.pow(tmp_image[:, :-1, :] - tmp_image[:, 1:, :], 2))
    img_tv_loss = (h_variance + w_variance) / 2.0
    
    '''gaussian reg'''
    loss_delta_opacity = torch.norm(gaussians._opacity - gt_gaussians._opacity.clone().detach())
    loss_delta_scaling = torch.norm(gaussians._scaling - gt_gaussians._scaling.clone().detach())
    loss_delta_rotation = torch.norm(gaussians._rotation - gt_gaussians._rotation.clone().detach())
    
    return  content_loss * opt.lambda_con + global_depth_reg * opt.lambda_depth + img_tv_loss * opt.lambda_tv + \
            loss_delta_opacity * opt.lambda_opacity + loss_delta_scaling * opt.lambda_scale + loss_delta_rotation * opt.lambda_rotation

def compute_geometry_loss(render_pkg, nnfm_loss_contra_fn, resized_image, style_depth, style_edge, opt) :
    
    '''render edge extract'''      
    canny_low = 25 ; canny_high = 200
    image = render_pkg["render"]
    image_np = image.squeeze(0).detach().cpu().numpy()
    image_np = np.transpose(image_np, (1, 2, 0))
    image_normalized = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    image_uint8 = (image_normalized * 255).astype(np.uint8)

    image_edge = []
    for i in range(3):
        edges = cv2.Canny(image_uint8[:, :, i], canny_low, canny_high)
        image_edge.append(edges)
    image_edge = np.stack(image_edge, axis=-1)
    image_edge = np.mean(image_edge, axis=-1)
    
    image_edge = torch.from_numpy(image_edge).unsqueeze(0).unsqueeze(0).float() / 255.0
    image_edge = image_edge.squeeze(0)
    image_edge = torch.cat([image_edge]*3, dim = 0)
    image_edge = image_edge.to("cuda")
    
    '''GCFM Loss'''
    image = render_pkg["render"]
    image = image.cuda()
    
    depth = render_pkg["depth"]
    depth = depth.unsqueeze(0)
    
    d_min = depth.amin(dim=(2,3), keepdim=True)
    d_max = depth.amax(dim=(2,3), keepdim=True)
    depth = (depth - d_min) / (d_max - d_min + 1e-8)

    depth = depth.squeeze(0)
    depth = torch.cat([depth]*3, dim = 0)
    
    gcfm_loss = nnfm_loss_contra_fn(blocks = [2,], outputs=image.unsqueeze(0), styles=resized_image.unsqueeze(0), 
                            output_depth = depth.unsqueeze(0), style_depth = style_depth.unsqueeze(0),
                            output_edge = image_edge.unsqueeze(0), style_edge = style_edge.unsqueeze(0))
    
    return gcfm_loss * opt.lambda_gcfm

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar(
            'train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(
                        viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(
                            viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(
                                viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                    iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(
                        config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(
                'total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def get_features(original: torch.Tensor, downscale=True) -> torch.Tensor:
    image = original.unsqueeze(0)
    
    if downscale:
        image = F.interpolate(image, scale_factor=0.5, mode="bilinear")
    
    normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    feature_layers = [11, 13, 15]
    image = normalize(image)
    
    outputs = []
    final_layer = max(feature_layers)
    
    for idx, layer in enumerate(VGG.features):
        image = layer(image)
        if idx in feature_layers:
            outputs.append(image)
        if idx == final_layer:
            break
            
    return torch.cat(outputs, dim=1).squeeze()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+",
                        type=int, default=[1, 1_000, 5_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+",
                        type=int, default=[30_000, 31_000, 35_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations",
                        nargs="+", type=int, default=[])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--style_image", type=str, default=None)

    parser.add_argument("--ply_path", type=str, default=None)

    parser.add_argument("--object", action="store_true")
    parser.add_argument("--geometry_stylization", action="store_true", default=False)
    parser.add_argument("--debug_stylized", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args),
             op.extract(args),
             pp.extract(args),
             args)
    
    print("\nTraining complete.")
import torch
import torchvision
import torch.nn.functional as F

class NNFMLossContra(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.vgg = torchvision.models.vgg16(pretrained=True).eval().to(device)
        self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_feats(self, x, layers=[], downscale = True):
        if downscale :
            x = F.interpolate(x, scale_factor = 0.5, mode = "bilinear")
        
        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs

    def forward(
        self,
        outputs, styles,
        output_depth, style_depth,
        output_edge, style_edge,
        blocks=[2,]
    ):
        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        blocks.sort()
        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.get_feats(outputs, all_layers, downscale = True)
        output_depth_feats = self.get_feats(output_depth, all_layers, downscale = True)
        output_edge_feats = self.get_feats(output_edge, all_layers, downscale = True)
        with torch.no_grad():
            s_feats_all = self.get_feats(styles, all_layers, downscale = False)
            style_depth_feats = self.get_feats(style_depth, all_layers, downscale = False)
            style_edge_feats = self.get_feats(style_edge, all_layers, downscale = False)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        gcfm_loss = 0
        for block in blocks:
            layers = block_indexes[block]

            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)
            output_depth_f = torch.cat([output_depth_feats[ix_map[ix]] for ix in layers], 1)
            style_depth_f = torch.cat([style_depth_feats[ix_map[ix]] for ix in layers], 1)
            output_edge_f = torch.cat([output_edge_feats[ix_map[ix]] for ix in layers], 1)
            style_edge_f = torch.cat([style_edge_feats[ix_map[ix]] for ix in layers], 1)
            
            target_feats, target_depth_feats, target_edge_feats, worst_feats, worst_depth_feats, worst_edge_feats = nn_feat_replace(x_feats, s_feats, 
                                                                                                                        output_depth_f, style_depth_f, 
                                                                                                                        output_edge_f, style_edge_f)
            gcfm_loss += cos_loss(x_feats, target_feats)
            gcfm_loss += cos_loss(output_depth_f, target_depth_feats)
            gcfm_loss += cos_loss(output_edge_f, target_edge_feats)
            
            gcfm_loss += neg_cos_loss(x_feats, worst_feats)
            gcfm_loss += neg_cos_loss(output_depth_f, worst_depth_feats)
            gcfm_loss += neg_cos_loss(output_edge_f, worst_edge_feats)

        return gcfm_loss
    
def argmin_cos_distance(a, b):
    b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)

    z_best = []
    z_worst = []
    loop_batch_size = int(1e8 / b.shape[-1])
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdims=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)

        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)

        z_best_batch = torch.argmin(d_mat, 2)
        z_worst_batch = torch.argmax(d_mat, 2)
        z_best.append(z_best_batch)
        z_worst.append(z_worst_batch)
    z_best = torch.cat(z_best, dim=-1)
    z_worst = torch.cat(z_worst, dim=-1)

    return z_best, z_worst

def nn_feat_replace(a, b, render_depth,  style_depth, render_edge, style_edge):
    n, c, h, w = a.size()
    n2, c, _, _ = b.size()
    n3, c, _, _ = style_depth.size()
    n4, c, _, _ = style_edge.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    r_flat = render_depth.view(n, c, -1)
    s_flat = style_depth.view(n3, c, -1)
    re_flat = render_edge.view(n, c, -1)
    se_flat = style_edge.view(n4, c, -1)
    
    b_ref = b_flat.clone()
    s_ref = s_flat.clone()
    se_ref = se_flat.clone()

    joint_content = torch.cat([a_flat, r_flat, re_flat], dim=1)
    joint_style = torch.cat([b_flat, s_flat, se_flat], dim=1)

    z_new = []
    s_new = []
    e_new = []
    zw_new = []
    sw_new = []
    ew_new = []
    for i in range(n):
        z_best, z_worst = argmin_cos_distance(joint_content[i : i + 1], joint_style[i : i + 1])
        z_best = z_best.unsqueeze(1).repeat(1, c, 1)
        z_worst = z_worst.unsqueeze(1).repeat(1, c, 1)
        
        feat = torch.gather(b_ref, 2, z_best)
        z_new.append(feat)
        
        d_feat = torch.gather(s_ref, 2, z_best)
        s_new.append(d_feat)
        
        e_feat = torch.gather(se_ref, 2, z_best)
        e_new.append(e_feat)

        feat = torch.gather(b_ref, 2, z_worst)
        zw_new.append(feat)
        
        d_feat = torch.gather(s_ref, 2, z_worst)
        sw_new.append(d_feat)
        
        ew_feat = torch.gather(se_ref, 2, z_worst)
        ew_new.append(ew_feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    
    s_new = torch.cat(s_new, 0)
    s_new = s_new.view(n, c, h, w)

    e_new = torch.cat(e_new, 0)
    e_new = e_new.view(n, c, h, w)
    
    zw_new = torch.cat(zw_new, 0)
    zw_new = zw_new.view(n, c, h, w)
    
    sw_new = torch.cat(sw_new, 0)
    sw_new = sw_new.view(n, c, h, w)

    ew_new = torch.cat(ew_new, 0)
    ew_new = ew_new.view(n, c, h, w)
    
    return z_new, s_new, e_new, zw_new, sw_new, ew_new

def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()

def neg_cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 + cossim
    return cos_d.mean()
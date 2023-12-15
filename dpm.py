import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab.modules.common.geometry import apply_rotation_to_vector, quaternion_1ijk_to_rotation_matrix
from diffab.modules.common.so3 import so3vec_to_rotation, rotation_to_so3vec, random_uniform_so3
from diffab.modules.diffusion.transition import RotationTransition, PositionTransition, AminoacidCategoricalTransition

import egnn_clean as eg


def rotation_matrix_cosine_loss(R_pred, R_true):
    """
    Args:
        R_pred: (*, 3, 3).
        R_true: (*, 3, 3).
    Returns:
        Per-matrix losses, (*, ).
    """
    size = list(R_pred.shape[:-2])
    ncol = R_pred.numel() // 3

    RT_pred = R_pred.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)
    RT_true = R_true.transpose(-2, -1).reshape(ncol, 3) # (ncol, 3)

    ones = torch.ones([ncol, ], dtype=torch.long, device=R_pred.device)
    loss = F.cosine_embedding_loss(RT_pred, RT_true, ones, reduction='none')  # (ncol*3, )
    loss = loss.reshape(size + [3]).sum(dim=-1)    # (*, )
    return loss


class FullDPM(nn.Module):

    def __init__(
        self, 
        in_node_nf =26, 
        hidden_nf=26,
        out_node_nf=23,
        num_steps=100, 
        n_layers=4, 
        trans_rot_opt={}, 
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_mean=[0.0, 0.0, 0.0],
        position_scale=[10.0],
    ):
        super().__init__()
        eps_net_opt={"attention":True, "normalize":True,"n_layers":n_layers}
        self.eps_net = eg.EGNN(in_node_nf=in_node_nf, hidden_nf=hidden_nf, out_node_nf=out_node_nf, **eps_net_opt)
        self.num_steps = num_steps
        self.trans_rot = RotationTransition(num_steps, **trans_rot_opt)
        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_mean', torch.FloatTensor(position_mean).view(1, 1, -1))
        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p - self.position_mean) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale + self.position_mean
        return p

    def forward(self, p_0, c_0, v_0, e, t=None):
        # L is sequence length, N is 1
        # p_0: (N, L, 3) coordinates
        # c_0: (N, L, 20) one-hot encoding of amino acid sequence
        # v_0: (N, L, 3) SO(3) vector of orientations
        # e: [(N, L), (N, L)] edge list
        edges=[]
        for edge in e:
            edges.append(edge.squeeze(0))
        # p_0=p_0.unsqueeze(0) # N,L,3 , N=1
        # c_0=c_0.unsqueeze(0) # N,L,20
        # v_0=v_0.unsqueeze(0) # N,L,3
    

        N, L = p_0.shape[:2]

        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)

        mask_generate = torch.full((N,L), True, dtype=torch.bool, device = p_0.device) #or 0s?

        # Normalize positions
        p_0 = self._normalize_position(p_0)

        #          As the coordinate deviation are
        # calculated in the local frame, we left-multiply it by the orientation matrix and transform it back to the
        # global frame [28]. Formally, this can be expressed as ϵˆj = Ot
        # j MLPG (hj ). Predicting coordinate
        # deviations in the local frame and projecting it to the global frame ensures the equivariance of the
        # prediction [28], as when the entire 3D structure rotates by a particular angle, the coordinate deviations
        # also rotates the same angle.
        # local_to_global

        # Add noise to rotation
        R_0 = so3vec_to_rotation(v_0)
        v_noisy, _ = self.trans_rot.add_noise(v_0, mask_generate, t)
        R_noisy = so3vec_to_rotation(v_noisy) # (N, L, 3, 3)

        # Add noise to positions
        p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
    
        # Add noise to sequence
        s_0=self.trans_seq._sample(c_0) # c_0 should be N,L,20
        c_noisy, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        
        beta = self.trans_pos.var_sched.betas[t]

        p_noisy = p_noisy.squeeze(0) # L,3
        v_noisy = v_noisy.squeeze(0) # L,3
        c_noisy = c_noisy.squeeze(0) # L,20
        
        p_0=p_0.squeeze(0) # L,3
        c_0=c_0.squeeze(0) # L,20
        v_0=v_0.squeeze(0) # L,3
    
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].squeeze(0).expand(L, 3) # (L, 3)
            
        in_feat=torch.cat((c_noisy, v_noisy, t_embed), dim=1) # (L x 26)

        pred_node_feat , p_pred = self.eps_net(h=in_feat, x=p_noisy, edges=edges, edge_attr=None) #(L x 23), (L x 3)
        
        c_denoised = pred_node_feat[:, :20]
        eps_rot = pred_node_feat[:, 20:]
        
        # Softmax
        c_denoised = F.softmax(c_denoised, dim=-1)

        eps_rot = eps_rot.unsqueeze(0)  # (1, L, 3)
        
        c_denoised = c_denoised.unsqueeze(0) # (1, L, 20)

        v_noisy = v_noisy.unsqueeze(0) # 1,L,3
        p_noisy = p_noisy.unsqueeze(0) # 1,L,3
        p_pred = p_pred.unsqueeze(0) # 1,L,3

        # from protdiff
        eps_p_pred = p_pred - p_noisy

        # New orientation
        U = quaternion_1ijk_to_rotation_matrix(eps_rot) # (N, L, 3, 3)
        R_pred = R_noisy @ U

        loss_dict = {}

        # Rotation loss
        loss_rot = rotation_matrix_cosine_loss(R_pred, R_0) # (N, L)
        loss_rot = (loss_rot * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['rot'] = loss_rot

        # Position loss
        loss_pos = F.mse_loss(eps_p_pred, eps_p, reduction='none').sum(dim=-1)  # (N, L, 3)
        # loss_pos = F.mse_loss(p_pred, p_0, reduction='none').sum(dim=-1)  # (L, 3)
        loss_pos = (loss_pos * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['pos'] = loss_pos

        # Sequence categorical loss
        post_true = self.trans_seq.posterior(s_noisy, s_0, t)
        log_post_pred = torch.log(self.trans_seq.posterior(s_noisy, c_denoised, t) + 1e-8)
        kldiv = F.kl_div(
            input=log_post_pred, 
            target=post_true, 
            reduction='none',
            log_target=False
        ).sum(dim=-1)    # (N, L)
        loss_seq = (kldiv * mask_generate).sum() / (mask_generate.sum().float() + 1e-8)
        loss_dict['seq'] = loss_seq

        return loss_dict

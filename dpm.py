import torch
import torch.nn as nn
import torch.nn.functional as F

from diffab.modules.diffusion.transition import PositionTransition, AminoacidCategoricalTransition

import egnn_complex as eg


class FullDPM(nn.Module):

    def __init__(
        self, 
        in_node_nf =23, 
        hidden_nf=20,
        out_node_nf=20,
        num_steps=100, 
        n_layers=4, 
        x_dim =9,
        additional_layers=0,
        trans_pos_opt={}, 
        trans_seq_opt={},
        position_scale=[10.0],
    ):
        super().__init__()
        
        self.eps_net = eg.EGNN(in_node_nf=in_node_nf, hidden_nf=hidden_nf, out_node_nf=out_node_nf,x_dim=x_dim,attention=True,n_layers=n_layers,additional_layers=additional_layers)
        self.num_steps = num_steps

        self.trans_pos = PositionTransition(num_steps, **trans_pos_opt)
        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('position_scale', torch.FloatTensor(position_scale).view(1, 1, -1))
        self.register_buffer('_dummy', torch.empty([0, ]))

    def _normalize_position(self, p):
        p_norm = (p) / self.position_scale
        return p_norm

    def _unnormalize_position(self, p_norm):
        p = p_norm * self.position_scale
        return p

    def forward(self, p_0, c_0, v_0, e, t=None,analyse=False):
        # L is sequence length, N is 1
        # p_0: (N, L, 3) coordinates
        # c_0: (N, L, 20) one-hot encoding of amino acid sequence
        # e: [(N, L), (N, L)] edge list
        edges=[]
        for edge in e:
            edges.append(edge.squeeze(0))
        # p_0=p_0.unsqueeze(0) # N,L,3 , N=1
        # c_0=c_0.unsqueeze(0) # N,L,20
    

        N, L = p_0.shape[:2]

        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)

        mask_generate = torch.full((N,L), True, dtype=torch.bool, device = p_0.device) #or 0s?

        # Normalize positions
        p_0 = self._normalize_position(p_0)

        # Add noise to positions
        p_noisy, eps_p = self.trans_pos.add_noise(p_0, mask_generate, t)
    
        # Add noise to sequence
        s_0=self.trans_seq._sample(c_0) # c_0 should be N,L,20
        c_noisy, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)
        
        beta = self.trans_pos.var_sched.betas[t]

        p_noisy = p_noisy.squeeze(0) # L,3
        
        c_noisy = c_noisy.squeeze(0) # L,20
        
        p_0=p_0.squeeze(0) # L,3
        c_0=c_0.squeeze(0) # L,20
        
    
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].squeeze(0).expand(L, 3) # (L, 3)
            
        pred_node_feat , eps_pred = self.eps_net(h=c_noisy, x=p_noisy.clone().detach(), t=t_embed, edges=edges, edge_attr=None) #(L x 23), (L x 3)
        

        c_denoised = pred_node_feat[:, :20]
        
        
        # Softmax
        c_denoised = F.softmax(c_denoised, dim=-1)        
        c_denoised = c_denoised.unsqueeze(0) # (1, L, 20)

        # print(eps_p.shape, eps_pred.shape)

        p_noisy = p_noisy.unsqueeze(0) # 1,L,3
        # p_pred = p_pred.unsqueeze(0) # 1,L,3
        eps_pred = eps_pred.unsqueeze(0) # 1,L,3
        eps_pred = eps_pred - p_noisy
        p_0 = p_0.unsqueeze(0) # 1,L,3


        loss_dict = {}

        # Position loss
        loss_pos = F.mse_loss(eps_pred, eps_p, reduction='none').sum(dim=-1)  # (L, 3)
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
        if analyse:
            return loss_dict, p_pred, p_0, c_0, c_denoised    
        return loss_dict

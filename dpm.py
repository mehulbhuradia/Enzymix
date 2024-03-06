import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from tqdm.auto import tqdm
from diffab.modules.diffusion.transition import PositionTransition, AminoacidCategoricalTransition
from diffab.modules.common.layers import clampped_one_hot
from egnn_clean import EGNN
# from gacnn import GACNN

class FullDPM(nn.Module):

    def __init__(
        self, 
        in_node_nf =23, 
        hidden_nf=23,
        out_node_nf=20,
        num_steps=100, 
        n_layers=4, 
        additional_layers=24,
        trans_seq_opt={},
    ):
        super().__init__()
        
        self.eps_net = EGNN(in_node_nf=in_node_nf, hidden_nf=hidden_nf, out_node_nf=out_node_nf,n_layers=n_layers,additional_layers=additional_layers)
        self.num_steps = num_steps

        self.trans_seq = AminoacidCategoricalTransition(num_steps, **trans_seq_opt)

        self.register_buffer('_dummy', torch.empty([0, ]))
    

    def forward(self, c_0, e, t=None,analyse=False):
        # L is sequence length, N is 1
        # p_0: (N, L, 3) coordinates
        # c_0: (N, L, 20) one-hot encoding of amino acid sequence
        # e: [(N, L), (N, L)] edge list
        
        edges=[]
        for edge in e:
            edges.append(edge.squeeze(0))
        
        N, L = c_0.shape[:2]

        if t == None:
            t = torch.randint(0, self.num_steps, (N,), dtype=torch.long, device=self._dummy.device)

        mask_generate = torch.full((N,L), True, dtype=torch.bool, device = c_0.device) 

    
        # Add noise to sequence
        s_0=self.trans_seq._sample(c_0) # c_0 should be N,L,20
        c_noisy, s_noisy = self.trans_seq.add_noise(s_0, mask_generate, t)

        beta = self.trans_seq.var_sched.betas[t]
        
        c_noisy = c_noisy.squeeze(0) # L,20
        
        c_0=c_0.squeeze(0) # L,20
    
        t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].squeeze(0).expand(L, 3) # (L, 3)
        
        c_denoised = self.eps_net(h=c_noisy,  t=t_embed, edges=edges) 
        
        # Softmax
        c_denoised = F.softmax(c_denoised, dim=-1)        
        c_denoised = c_denoised.unsqueeze(0) # (1, L, 20)
        
        loss_dict = {}

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
            return loss_dict, c_0, c_denoised, t
        return loss_dict
  
    
    @torch.no_grad()
    def sample(
        self, 
        c, 
        e
    ):
        # L is sequence length, N is 1
        # c_0: (N, L, 20) one-hot encoding of amino acid sequence
        # e: [(N, L), (N, L)] edge list

        edges=[]
        for edge in e:
            edges.append(edge.squeeze(0))
        

        N, L = c.shape[:2]

        mask_generate = torch.full((N,L), True, dtype=torch.bool, device = c.device) #or 0s?
         
        s=self.trans_seq._sample(c) # c_0 should be N,L,20

        s_rand = torch.randint_like(s, low=0, high=19)
        s_init = torch.where(mask_generate, s_rand, s)
    

        traj = {self.num_steps: s_init}
        
        pbar = functools.partial(tqdm, total=self.num_steps, desc='Sampling')
        
        for t in pbar(range(self.num_steps, 0, -1)):
            s_t = traj[t]
            
            t_tensor = torch.full([N, ], fill_value=t, dtype=torch.long, device=self._dummy.device)

            beta = self.trans_seq.var_sched.betas[t].expand([N, ])
            t_embed = torch.stack([beta, torch.sin(beta), torch.cos(beta)], dim=-1)[:, None, :].squeeze(0).expand(L, 3) # (L, 3)
            
            c_t = clampped_one_hot(s_t, num_classes=20).float() # (N, L, K).

            
            c_t=c_t.squeeze(0) # L,20        
    
            c_denoised = self.eps_net(h=c_t,  t=t_embed, edges=edges) 

            # Softmax
            c_denoised = F.softmax(c_denoised, dim=-1)        
            c_denoised = c_denoised.unsqueeze(0) # (1, L, 20)

            _, s_next = self.trans_seq.denoise(s_t, c_denoised, mask_generate, t_tensor)

            traj[t-1] = s_next
            traj[t] = tuple(x.cpu() for x in traj[t])    # Move previous states to cpu memory.

        return traj
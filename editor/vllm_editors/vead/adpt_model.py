#%%
from typing import Dict, List, Tuple
from torch import nn
import torch
import numpy as np

GATING_STATUS = False
LAST_MODALITY = None
BATCH_SIM_LIST = None
THREHSHOLD = 0.8

class BaseResMapper(nn.Module):
    def __init__(self, in_dim, mid_dim, act_layer = nn.ReLU):
        super().__init__()
        self.l_in = nn.Linear(in_dim, mid_dim)
        self.act = act_layer()
        self.l_out = nn.Linear(mid_dim, in_dim)
        
    def reset_parameters(self):
        self.l_in.reset_parameters()
        self.l_out.reset_parameters()

    def forward(self, x):
        return self.l_out(self.act(self.l_in(x))) + x
    
class InfluenceMapper(nn.Module):
    def __init__(self, inpt_dim, mid_dim, att_head_n):
        super().__init__()
        self.ln_img_reps = nn.LayerNorm(inpt_dim) 
        self.ln_edit_reps = nn.LayerNorm(inpt_dim) 
        self.img_map = BaseResMapper(inpt_dim, mid_dim)
        self.prompt_token_map = BaseResMapper(inpt_dim, mid_dim)
        self.att_head_n = att_head_n
        self.scale = (inpt_dim // att_head_n) ** 0.5
    
    def reset_parameters(self):
        self.ln_img_reps.reset_parameters()
        self.ln_edit_reps.reset_parameters()
        self.img_map.reset_parameters()
        self.prompt_token_map.reset_parameters()

    def forward(self, img_reps, prompt_last_token_of_edit_reps):
        '''`img_reps`: [b, img_token_n, d], prompt_last_token_of_edit_reps: [b, d]'''
        b, tn, d = img_reps.shape
        img_reps = self.img_map(self.ln_img_reps(img_reps)) # [b, img_token_n, d]
        prompt_last_token_of_edit_reps = self.prompt_token_map(self.ln_edit_reps(prompt_last_token_of_edit_reps))
        img_reps = img_reps.reshape(b, tn, self.att_head_n, d//self.att_head_n) # [b,img_token_n,head,d//head]
        prompt_last_token_of_edit_reps = prompt_last_token_of_edit_reps.reshape(b, self.att_head_n, d//self.att_head_n)
        inf_map = torch.einsum('bihd,bhd->bih', img_reps, prompt_last_token_of_edit_reps) # [b, img_token_n, d//head]
        inf_map = inf_map.mean(2) / self.scale # [b, img_token_n]
        return inf_map

class VisionEditAdaptor(nn.Module):
    def __init__(self, hidden_size, mid_dim = 1024, cross_att_head_n = 8, 
                 img_tok_n = 576, add_it = False, infm_dim = 256) -> None:
        '''
        hidden_size: dimension of embeddings
        mid_dim: middle dimension of adaptor
        cross_att_head_n: number of cross attention heads
        '''
        super().__init__()
        if mid_dim % cross_att_head_n != 0: raise
        self.mid_dim = mid_dim
        self.cross_att_head_n = cross_att_head_n
        self.add_it = add_it
        self.img_tok_n = img_tok_n
        self.mlp_begin = nn.Linear(hidden_size, mid_dim)
        self.cross_att_q_mlp = nn.Linear(mid_dim, mid_dim)
        self.cross_att_k_mlp = nn.Linear(hidden_size, mid_dim)
        self.cross_att_v_mlp = nn.Linear(hidden_size, mid_dim)
        self.mlp_end = nn.Linear(mid_dim, hidden_size)
        self.ln_img_reps = nn.LayerNorm(hidden_size) 
        self.ln_edit_reps = nn.LayerNorm(hidden_size) 
        self.influence_mapper = InfluenceMapper(hidden_size, infm_dim, cross_att_head_n) 
        self.open_gating = False
        self.open_adaptor(False)
        self.set_edit_signal(None, None, None)

    def reset_parameters(self):
        self.mlp_begin.reset_parameters()
        self.cross_att_q_mlp.reset_parameters()
        self.cross_att_k_mlp.reset_parameters()
        self.cross_att_v_mlp.reset_parameters()
        self.mlp_end.reset_parameters()
        self.ln_img_reps.reset_parameters()
        self.ln_edit_reps.reset_parameters()
        self.influence_mapper.reset_parameters()

    def forward(self, layer_outpt):
        '''layer_outpt: [b, l, d]''' 
        if (not self.is_open 
            or layer_outpt.shape[1] == 1 # generate mode, which has attention cache 
            or not self.inpt_has_img # no vision token in this input
            ): return layer_outpt
        
        layer_input = layer_outpt.clone()

        global GATING_STATUS
        global BATCH_SIM_LIST
        global THREHSHOLD
        global LAST_MODALITY

        if LAST_MODALITY != "image":
            GATING_STATUS = not GATING_STATUS
            LAST_MODALITY = 'image'
            if GATING_STATUS:
                similarity = torch.nn.functional.cosine_similarity(layer_outpt[:,-1,:], self.edit_reps[:,-1,:], dim=-1)
                BATCH_SIM_LIST = similarity 
                condition = (BATCH_SIM_LIST < THREHSHOLD).unsqueeze(-1).unsqueeze(1).expand(layer_input.size())
            else:
                condition = (BATCH_SIM_LIST < THREHSHOLD).unsqueeze(-1).unsqueeze(1).expand(layer_input.size())
        else:
            similarity = torch.nn.functional.cosine_similarity(layer_outpt[:,-1,:], self.edit_reps[:,-1,:], dim=-1)
            BATCH_SIM_LIST = similarity 
            condition = (BATCH_SIM_LIST < THREHSHOLD).unsqueeze(-1).unsqueeze(1).expand(layer_input.size())

        if self.inpt_vt_begin == None or self.inpt_vt_end == None:  
            raise BaseException('Have not set vision token range.')
        # get normed reps and determine validity
        img_reps = layer_outpt[:, self.inpt_vt_begin:self.inpt_vt_end].clone()
        b1, l1, _ = img_reps.shape # l1 = self.img_tok_n
        b2, l2, _ = self.edit_reps.shape 
        if l1 != self.img_tok_n: 
            raise BaseException('Number of selected vision tokens error.')
        if b1 != b2: 
            raise BaseException('Batch size of input and editing signal are not matched.')
        # introduce influence mapping
        if self.add_it:
            prompt_last_token_of_edit_reps = self.edit_reps[range(len(self.prompt_end)), self.prompt_end] # [b, d]
            inf_map = self.influence_mapper(img_reps, prompt_last_token_of_edit_reps) # [b, img_token_n]
            inf_map = torch.sigmoid(inf_map).unsqueeze(-1) # [b, img_token_n, 1]
        else:
            inf_map = 1
        # image representation transformation
        norm_img_reps = self.ln_img_reps(img_reps)  # [b,img_tok_n,d]
        norm_edit_reps = self.ln_edit_reps(self.edit_reps) # [b,l,d]
        x = self.mlp_begin(norm_img_reps)
        q = self.cross_att_q_mlp(x).reshape(b1, l1, self.cross_att_head_n, self.mid_dim//self.cross_att_head_n)
        k = self.cross_att_k_mlp(norm_edit_reps).reshape(b1, l2, self.cross_att_head_n, self.mid_dim//self.cross_att_head_n)
        v = self.cross_att_v_mlp(norm_edit_reps).reshape(b1, l2, self.cross_att_head_n, self.mid_dim//self.cross_att_head_n)
        s = torch.einsum('blhm,buhm->bhlu', q, k) # [batch_size, head_n, l1, l2]
        s = s / (self.mid_dim//self.cross_att_head_n)**0.5
        s = s + (self.edit_reps_att_mask.reshape(b1, 1, 1, l2) - 1)*9999999999
        s = torch.softmax(s, 3)
        x = torch.einsum('bhlu,buhm->blhm', s, v).reshape(b1, l1, self.mid_dim) # [batch_size, img_tok_n, mid_dim]
        x = self.mlp_end(x) * inf_map
        layer_outpt[:, self.inpt_vt_begin:self.inpt_vt_end] = img_reps + x
        
        if self.open_gating:
            output = torch.where(condition, layer_input, layer_outpt)  # [b, l, d]
            layer_outpt = output

        return layer_outpt

    def open_adaptor(self, if_open:bool):
        self.is_open = if_open
     

    def set_edit_signal(self, edit_reps:torch.Tensor, edit_reps_att_mask:torch.Tensor, 
                           prompt_end:torch.Tensor):
        # edit_reps/edit_reps_att_mask: [b,l,d], prompt_end: [b]
        self.edit_reps = edit_reps
        self.edit_reps_att_mask = edit_reps_att_mask
        self.prompt_end = prompt_end
    
    def set_input_info(self, has_img = True, vt_begin:int = None, vt_end:int = None):
        '''Should be called every time input is fed to the LLM. '''
        self.inpt_has_img = has_img # whether the input has image
        self.inpt_vt_begin = vt_begin # start index of vision tokens
        self.inpt_vt_end = vt_end # end index of vision tokens
    
    def set_prompt_end(self, prompt_end:torch.Tensor):
        pass
    
class TextEditAdaptor(nn.Module):
    def __init__(self, hidden_size, mid_dim = 1024, cross_att_head_n = 8, add_it = False, infm_dim = 256) -> None:
        '''
        Adapter for editing text tokens
        hidden_size: embedding dimension
        mid_dim: middle dimension of adaptor
        cross_att_head_n: number of cross attention heads
        '''
        super().__init__()
        if mid_dim % cross_att_head_n != 0: raise
        self.mid_dim = mid_dim
        self.cross_att_head_n = cross_att_head_n
        self.add_it = add_it
        self.mlp_begin = nn.Linear(hidden_size, mid_dim)
        self.cross_att_q_mlp = nn.Linear(mid_dim, mid_dim)
        self.cross_att_k_mlp = nn.Linear(hidden_size, mid_dim)
        self.cross_att_v_mlp = nn.Linear(hidden_size, mid_dim)
        self.mlp_end = nn.Linear(mid_dim, hidden_size)
        self.ln_text_reps = nn.LayerNorm(hidden_size) 
        self.ln_edit_reps = nn.LayerNorm(hidden_size) 
        self.influence_mapper = InfluenceMapper(hidden_size, infm_dim, cross_att_head_n) if add_it else None
        self.open_adaptor(False)
        self.set_edit_signal(None, None, None)
        self.text_token_indices = None

    def reset_parameters(self):
        self.mlp_begin.reset_parameters()
        self.cross_att_q_mlp.reset_parameters()
        self.cross_att_k_mlp.reset_parameters()
        self.cross_att_v_mlp.reset_parameters()
        self.mlp_end.reset_parameters()
        self.ln_text_reps.reset_parameters()
        self.ln_edit_reps.reset_parameters()
        if self.influence_mapper is not None:
            self.influence_mapper.reset_parameters()

    def forward(self, layer_outpt):
        '''layer_outpt: [b, l, d] or list''' 
        # Record input type
        if not self.is_open:
            return layer_outpt
        is_list_input = isinstance(layer_outpt, list)
        if is_list_input:
            if len(layer_outpt) != 1:
                print(f"The length of layer_outpt should be 1, but got {len(layer_outpt)}")
            layer_outpt = layer_outpt[0]
            
        if (not self.is_open 
            or layer_outpt.shape[1] == 1 # generation mode, with attention cache
            ): 
            return [layer_outpt] if is_list_input else layer_outpt
        
        layer_input = layer_outpt.clone()

        global GATING_STATUS
        global BATCH_SIM_LIST
        global THREHSHOLD
        global LAST_MODALITY

        if LAST_MODALITY != 'text':
            GATING_STATUS = not GATING_STATUS
            LAST_MODALITY = 'text'
            if GATING_STATUS:
                similarity = torch.nn.functional.cosine_similarity(layer_outpt[:,-1,:], self.edit_reps[:,-1,:], dim=-1)
                BATCH_SIM_LIST = similarity 
                condition = (BATCH_SIM_LIST < THREHSHOLD).unsqueeze(-1).unsqueeze(1).expand(layer_input.size())
            else:
                condition = (BATCH_SIM_LIST < THREHSHOLD).unsqueeze(-1).unsqueeze(1).expand(layer_input.size())
        else:
            similarity = torch.nn.functional.cosine_similarity(layer_outpt[:,-1,:], self.edit_reps[:,-1,:], dim=-1)
            BATCH_SIM_LIST = similarity 
            condition = (BATCH_SIM_LIST < THREHSHOLD).unsqueeze(-1).unsqueeze(1).expand(layer_input.size())

        # Get text token representations
        batch_size = layer_outpt.shape[0]
        text_indices = []
        for i in range(batch_size):
            if self.prompt_end is None:
                raise BaseException("prompt_end is None")
            if self.inpt_vt_end is not None:
                if self.prompt_end.dim() == 0:  # 0D tensor
                    indices = list(range(self.inpt_vt_end, self.prompt_end.item()))
                else:  # multi-D tensor
                    indices = list(range(self.inpt_vt_end, self.prompt_end[i]))
                text_indices.append(indices)
            else:
                if self.prompt_end.dim() == 0:  # 0D tensor
                    indices = list(range(1, self.prompt_end.item()))
                else:  # multi-D tensor
                    indices = list(range(1, self.prompt_end[i]))
                text_indices.append(indices)
        
        if len(text_indices) > 0:
            # Handle variable number of text tokens for each sample in batch
            text_reps_list = []
            for i, indices in enumerate(text_indices):
                if len(indices) > 0:
                    if i >= layer_outpt.shape[0] or max(indices) >= layer_outpt.shape[1]:
                        import pdb; pdb.set_trace()
                    text_reps_list.append(layer_outpt[i, indices])
                else:
                    print("No text token to edit, return original layer_outpt")
                    import pdb; pdb.set_trace()
            
            if not text_reps_list:
                print("No text token to edit, return original layer_outpt")
                raise
                
            # Process each sample separately
            for i, indices in enumerate(text_indices):
                if len(indices) == 0:
                    raise "no text token to edit"
                    
                text_reps = layer_outpt[i, indices].unsqueeze(0)  # [1, num_tokens, d]
                b1, l1, _ = text_reps.shape
                b2, l2, _ = self.edit_reps.shape
                
                inf_map = 1
                
                # Text representation transformation
                norm_text_reps = self.ln_text_reps(text_reps)  # [1, num_tokens, d]
                norm_edit_reps = self.ln_edit_reps(self.edit_reps[i:i+1])  # [1, l, d]
                
                x = self.mlp_begin(norm_text_reps)
                q = self.cross_att_q_mlp(x).reshape(b1, l1, self.cross_att_head_n, self.mid_dim//self.cross_att_head_n)
                k = self.cross_att_k_mlp(norm_edit_reps).reshape(b1, l2, self.cross_att_head_n, self.mid_dim//self.cross_att_head_n)
                v = self.cross_att_v_mlp(norm_edit_reps).reshape(b1, l2, self.cross_att_head_n, self.mid_dim//self.cross_att_head_n)
                
                s = torch.einsum('blhm,buhm->bhlu', q, k)  # [1, head_n, l1, l2]
                s = s / (self.mid_dim//self.cross_att_head_n)**0.5
                s = s + (self.edit_reps_att_mask[i:i+1].reshape(b1, 1, 1, l2) - 1)*9999999999
                s = torch.softmax(s, 3)
                
                x = torch.einsum('bhlu,buhm->blhm', s, v).reshape(b1, l1, self.mid_dim)  # [1, num_tokens, mid_dim]
                x = self.mlp_end(x) * inf_map
                layer_outpt[i, indices] = text_reps.squeeze(0) + x.squeeze(0)

        if self.open_gating:
            output = torch.where(condition, layer_input, layer_outpt)  # [b, l, d]
            layer_outpt =  output
        
        return [layer_outpt] if is_list_input else layer_outpt

    def open_adaptor(self, if_open:bool):
        self.is_open = if_open

    def open_gating(self, open_gating:bool):
        self.open_gating = open_gating

    def set_edit_signal(self, edit_reps:torch.Tensor, edit_reps_att_mask:torch.Tensor, 
                           prompt_end:torch.Tensor):
        # edit_reps/edit_reps_att_mask: [b,l,d], prompt_end: [b]
        self.edit_reps = edit_reps
        self.edit_reps_att_mask = edit_reps_att_mask
        self.prompt_end = prompt_end
    
    def set_prompt_end(self, prompt_end:torch.Tensor):
        self.prompt_end = prompt_end
    
    def set_text_token_indices(self, text_token_indices):
        '''Set the indices of text tokens to edit'''
        self.text_token_indices = text_token_indices

    def set_input_info(self, has_img = True, vt_begin:int = None, vt_end:int = None):
        '''Should be called every time input is fed to the LLM. '''
        self.inpt_has_img = has_img # whether the input has image
        self.inpt_vt_begin = vt_begin
        self.inpt_vt_end = vt_end

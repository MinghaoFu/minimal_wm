import torch
import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from .flow_kl_loss import ConditionalFlowKLLoss
from .criterion import SymlogMSELoss

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        emb_decoder,
        post_concat_projection,
        predictor,
        alignment_projection=None,  # nn.Linear for state alignment
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        train_decoder_grad=True,
        projected_dim=64,
        predictor_dim=None,  # Will be computed as projected_dim + action_dim
        alignment_dim=None,  # Number of dimensions for InfoNCE alignment (hyperparameter)
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.emb_decoder = emb_decoder
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.train_decoder_grad = train_decoder_grad
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.alignment_dim = min(self.proprio_dim, projected_dim)
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)
        
        # Calculate concatenated dimension based on concat_dim
        if self.concat_dim == 1:
            # Feature concatenation: visual(384) + proprio(32) + action(16) = 432
            self.concat_emb_dim = self.encoder.emb_dim + self.proprio_dim + self.action_dim
        else:
            # Token concatenation: all tokens have same visual dimension (384)
            self.concat_emb_dim = self.encoder.emb_dim
        
        # Add projection layer after concatenation (configurable target dimension)  
        self.projected_dim = projected_dim
        # Projection input excludes action dimensions (visual + proprio only)
        self.projection_input_dim = self.encoder.emb_dim + self.proprio_dim  # 384 + 32 = 416
        self.post_concat_projection = post_concat_projection
        
        # Compute predictor dimension dynamically  
        self.predictor_dim = self.projected_dim + self.action_dim
        
        print(f"Concat emb_dim: {self.concat_emb_dim} (visual:{self.encoder.emb_dim} + proprio:{self.proprio_dim} + action:{self.action_dim})")
        print(f"Projection input: {self.projection_input_dim}D (visual:{self.encoder.emb_dim} + proprio:{self.proprio_dim})")
        print(f"After projection: {self.projected_dim}D (compressed visual+proprio)")
        print(f"Final predictor input: {self.predictor_dim}D ({self.projected_dim}D projected + {self.action_dim}D action)")
        print(f"InfoNCE alignment: {self.alignment_dim}D (first {self.alignment_dim} dims of projected features → state_dim from data)")

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * self.encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        #self.decoder_criterion = nn.MSELoss()
        self.decoder_criterion = nn.MSELoss()
        self.emb_decoder_criterion = nn.MSELoss() #SymlogMSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss() # zloss

        # Linear alignment loss for state supervision
        self.alignment_projection = alignment_projection  # nn.Linear: R^{alignment_dim} -> R^{state_dim}
        self.state_consistency_loss_weight = 1.0
        self.alignment_regularization = 1e-4  # L2 regularization on W
        
        # Conditional Flow KL divergence loss
        self.flow_kl_loss_weight = 1.0
        self.flow_kl_loss_enabled = False  # Will be set via config
        self.flow_kl_loss = ConditionalFlowKLLoss(
            z_dim=128,  # DINO feature dimension
            state_dim=7,  # Robomimic state dimension
            hidden_dim=256
        )

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        if self.proprio_encoder is not None:
            self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.emb_decoder.train(mode)
            self.decoder.train(mode)

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        if self.proprio_encoder is not None:
            self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z_concat = torch.cat(
                    [z_dct['visual'], z_dct['proprio'].unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                )  # (b, num_frames, num_patches + 2, 384)
            # Apply projection only to visual + proprio (exclude action): 384D -> 64D
            z_visual_proprio = torch.cat([z_dct['visual'], z_dct['proprio'].unsqueeze(2)], dim=2)
            z_projected = self.post_concat_projection(z_concat)  # (b, num_frames, num_patches + 1, 64)
            return z_projected, z_dct
        if self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            # lag_act_repeated = torch.zeros_like(act_repeated).to(act_repeated.device)
            # lag_act_repeated[:, 1:, :, :] = act_repeated[:, :-1, :, :]
            o = torch.cat([z_dct['visual'], proprio_repeated], dim=3) 
            z_projected = self.post_concat_projection(z_dct['visual'], proprio_repeated) 
            z = torch.cat([z_projected, act_repeated], dim=3) 
        return z, z_dct
    
    def encode_to_projected(self, obs, act):
        z_dct = self.encode_obs(obs)
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z_visual_proprio = torch.cat([z_dct['visual'], z_dct['proprio'].unsqueeze(2)], dim=2)
            z_projected = self.post_concat_projection(z_visual_proprio) 
            return {"projected": z_projected}
        elif self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            # lag_act_repeated = torch.zeros_like(act_repeated).to(act_repeated.device)
            # lag_act_repeated[:, 1:, :, :] = act_repeated[:, :-1, :, :]
            o = torch.cat([z_dct['visual'], proprio_repeated], dim=3) 
            z_projected = self.post_concat_projection(z_dct['visual'], proprio_repeated) 
            z = torch.cat([z_projected, act_repeated], dim=3) 
            return {"projected": z_projected}
        
    def encode_obs_projected(self, obs):
        z_dct = self.encode_obs(obs)
        if self.concat_dim == 0:
            z_visual_proprio = torch.cat([z_dct['visual'], z_dct['proprio'].unsqueeze(2)], dim=2)
            z_projected = self.post_concat_projection(z_visual_proprio) 
            return {"projected": z_projected}
        elif self.concat_dim == 1:
            proprio_tiled = repeat(z_dct['proprio'].unsqueeze(2), "b t 1 a -> b t f a", f=z_dct['visual'].shape[2])
            proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
            o = torch.cat([z_dct['visual'], proprio_repeated], dim=3) 
            z_projected = self.post_concat_projection(z_dct['visual'], proprio_repeated) 
            return {"projected": z_projected}
        
    def encode_act(self, act):
        act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
        return act
    
    def encode_proprio(self, proprio):
        proprio = self.proprio_encoder(proprio)
        return proprio

    def encode_obs(self, obs, project=False):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        visual = rearrange(visual, "b t ... -> (b t) ...")
        visual = self.encoder_transform(visual)
        visual_embs = self.encoder.forward(visual)
        visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

        proprio = obs['proprio']
        proprio_emb = self.encode_proprio(proprio)
        return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        z_projected = z[:, :, :, :self.projected_dim]
        z_action = z[:, :, :, self.projected_dim:]
        T = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        # (b, num_hist * num_patches per img, emb_dim)
        z = self.predictor(z)
        z = rearrange(z, "b (t p) d -> b t p d", t=T)
        return z

    # def decode(self, z_concat):
    #     """
    #     input :   z: (b, num_frames, num_patches, emb_dim) - should be 64D projected features
    #     output: obs: (b, num_frames, 3, img_size, img_size)
        
    #     """
    #     b, num_frames, num_patches, concat_dim = z_concat.shape
        
    #     z_projected = z_concat[:, :, :, :self.projected_dim]
    #     z_action = z_concat[:, :, :, self.projected_dim:]
        
    #     visual, diff = self.decoder(z_projected) 
    #     visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
    #     obs = {
    #         "visual": visual,
    #         "action": z_action
    #     }
    
    #     return obs, diff
    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        z_obs = self.separate_s_a(z)
        obs, diff = self.decode_obs(z_obs)
        return obs, diff

    # def decode_obs(self, z_obs):
    #     """
    #     input :   z: (b, num_frames, num_patches, emb_dim) - 80D features [64D projected + 16D action]
    #     output: obs: (b, num_frames, 3, img_size, img_size)
    #     """
    #     b, num_frames, num_patches, emb_dim = z_obs["projected"].shape
    #     visual, diff = self.decoder(z_obs["projected"])  
    #     visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
    #     obs = {
    #         "visual": visual
    #     }
    #     return obs, diff
    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        b, num_frames, num_patches, emb_dim = z_obs["projected"].shape
        visual, diff = self.decoder(z_obs["projected"])  # (b*num_frames, 3, 224, 224)
        visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
        obs = {
            "visual": visual,
            # "proprio": z_obs["proprio"], 
        }
        return obs, diff
    
    # def separate_emb(self, z):
    #     """
    #     input: z (tensor)
    #     output: z_obs (dict), z_act (tensor)
    #     """
    #     if self.concat_dim == 0:
    #         z_projected, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
    #     elif self.concat_dim == 1:
    #         z_projected, z_act = z[..., :-self.action_dim], z[..., -self.action_dim:]
    #         # remove tiled dimensions
    #         z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
    #     z_obs = {"projected": z_projected}
    #     return z_obs, z_act
    # def separate_emb(self, z):
    #     """
    #     input: z (tensor)
    #     output: z_obs (dict), z_act (tensor)
    #     """
    #     if self.concat_dim == 0:
    #         z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
    #     elif self.concat_dim == 1:
    #         z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
    #                                      z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
    #                                      z[..., -self.action_dim:]
    #         # remove tiled dimensions
    #         z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
    #         z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
    #     z_obs = {"visual": z_visual, "proprio": z_proprio}
    #     return z_obs, z_act
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), 
        """
        if self.concat_dim == 0:
            z_visual, z_proprio = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio = z[..., :-self.proprio_dim], \
                                         z[..., -self.proprio_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs
    
    def separate_s_a(self, z):
        """
        input: z_pred (tensor)
        output: state_pred (tensor), z_tgt (tensor)
        """
        state_pred = z[:, :, :, :self.projected_dim]
        action_pred = z[:, :, :, self.projected_dim:]
        return {'projected': state_pred, 'action': action_pred}

    def infonce_loss(self, z_aligned, z_target, temperature=0.1):
        z_aligned_flat = z_aligned.reshape(-1, z_aligned.shape[-1])  # (b*num_hist, 7)
        z_target_flat = z_target.reshape(-1, z_target.shape[-1])    # (b*num_hist, 7)
        z_aligned_norm = torch.nn.functional.normalize(z_aligned_flat, dim=1)
        z_target_norm = torch.nn.functional.normalize(z_target_flat, dim=1)
        logits = torch.matmul(z_aligned_norm, z_target_norm.T) / temperature  # (N, N)
        labels = torch.arange(logits.shape[0], device=logits.device)
        
        loss_ce = torch.nn.functional.cross_entropy(logits, labels)
        return loss_ce
                
    def forward(self, obs, act, state=None):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
                state: (b, num_frames, state_dim) - true state variables for consistency loss
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        loss = 0
        loss_components = {}
        
        z, z_dct = self.encode(obs, act)  
        
        z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, 64)
        z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, 64)
        
        visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
        visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)
        
        if self.predictor is not None:
            z_pred = self.predict(z_src)
            if self.decoder is not None:
                b, num_frames, num_patches, emb_dim = z_pred.shape
                z_pred_emb = self.emb_decoder(self.separate_s_a(z_pred)['projected'])
                if self.train_decoder_grad:
                    obs_pred, diff_pred = self.decode(self.separate_s_a(z_pred)['projected'])
                else:
                    obs_pred, diff_pred = self.decode(self.separate_s_a(z_pred)['projected'].detach())
                visual_pred = obs_pred['visual']
                recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                emb_recon_loss_pred = self.emb_decoder_criterion(self.separate_emb(z_pred_emb)['visual'], z_dct['visual'][:, self.num_hist :, ...])
                decoder_loss_pred = (
                    recon_loss_pred + emb_recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                )
                loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                loss_components["emb_decoder_recon_loss_pred"] = emb_recon_loss_pred
                loss_components["decoder_vq_loss_pred"] = diff_pred
                loss_components["decoder_loss_pred"] = decoder_loss_pred
            else:
                visual_pred = None

            z_pred_projected = self.separate_s_a(z_pred)['projected']  
            z_tgt_projected = self.separate_s_a(z_tgt)['projected']    
            z_loss = self.emb_criterion(z_pred_projected, z_tgt_projected.detach())

            loss = loss + z_loss + decoder_loss_pred
            loss_components["z_predicted_loss"] = z_loss
            
            # InfoNCE alignment approach: first alignment_dim of projected features → actual state dimension
            if state is not None:
                if self.concat_dim == 0:
                    z_visual_for_state = z_pred[:, :, :-2, :]  # (b, num_hist, num_patches, 80D)
                elif self.concat_dim == 1:
                    z_visual_for_state = z_pred[:, :, :, :self.projected_dim]  # (b, num_hist, num_patches, 64D)
                
                z_avg = z_visual_for_state.mean(dim=2)  # (b, num_hist, z_dim)
                z_align_dims = z_avg[:, :, :self.alignment_dim]  # (b, num_hist, alignment_dim) - first alignment_dim dims
                z_target = state[:, self.num_pred:, :]  # (b, num_hist, state_dim)

                # Use alignment projection if available
                if self.alignment_projection is not None:
                    # Reshape for nn.Linear: (b*num_hist, alignment_dim) -> (b*num_hist, state_dim)
                    b, num_hist, align_dim = z_align_dims.shape
                    z_aligned = self.alignment_projection(z_align_dims.reshape(-1, align_dim))
                    z_aligned = z_aligned.reshape(b, num_hist, -1)  # (b, num_hist, state_dim)

                    alignment_loss = self.infonce_loss(z_aligned, z_target)

                    # L2 regularization on projection weights
                    w_regularization = self.alignment_regularization * torch.sum(self.alignment_projection.weight**2)
                else:
                    # Skip alignment if projection not available
                    alignment_loss = torch.tensor(0.0, device=z_align_dims.device)
                    w_regularization = torch.tensor(0.0, device=z_align_dims.device)
                
                # Total state consistency loss
                state_consistency_loss = alignment_loss + w_regularization
                loss = loss + self.state_consistency_loss_weight * state_consistency_loss
                loss_components["state_consistency_loss"] = state_consistency_loss
                loss_components["alignment_loss"] = alignment_loss
                loss_components["w_regularization"] = w_regularization

                if hasattr(self, 'latent_dynamics_loss_weight') and self.latent_dynamics_loss_weight > 0: # not used
                    if self.concat_dim == 0:
                        z_visual_src = z_src[:, :, :-2, :] 
                        z_visual_src = z_visual_src[:, :, :, :self.projected_dim] 
                    elif self.concat_dim == 1:
                        z_visual_src = z_src[:, :, :, :self.projected_dim]  
                    
                    z_src_avg = z_visual_src.mean(dim=2) 
                    z_pred_avg = z_avg  
                    
                    # Predict t from t-1 using 64D projected features (64D → 64D)
                    dynamics_loss = torch.mean((z_pred_avg - z_src_avg)**2)
                    loss = loss + self.latent_dynamics_loss_weight * dynamics_loss
                    loss_components["dynamics_projected_loss"] = dynamics_loss

            
            # DINO feature reconstruction loss (384D -> 128D -> 384D)
            if hasattr(self.encoder, 'recon_dino_loss') and self.encoder.recon_dino_loss: # not used
                # Get DINO reconstruction loss from encoder
                dino_recon_loss = getattr(self.encoder, 'last_recon_loss', None)
                if dino_recon_loss is not None and hasattr(self, 'dino_recon_loss_weight'):
                    loss = loss + self.dino_recon_loss_weight * dino_recon_loss
                    loss_components["dino_recon_loss"] = dino_recon_loss
            
        else:
            visual_pred = None
            z_pred = None

        if self.decoder is not None:
            # Following original DINO WM: pass full features to decoder
            z_emb_reconstructed = self.emb_decoder(self.separate_s_a(z)['projected'])
            if self.train_decoder_grad:
                obs_reconstructed, diff_reconstructed = self.decode(self.separate_s_a(z)['projected'])
            else:
                obs_reconstructed, diff_reconstructed = self.decode(self.separate_s_a(z)['projected'].detach())
            visual_reconstructed = obs_reconstructed["visual"]
            recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
            emb_recon_loss_reconstructed = self.emb_decoder_criterion(self.separate_emb(z_emb_reconstructed)['visual'], z_dct['visual'])
            decoder_loss_reconstructed = (
                recon_loss_reconstructed + emb_recon_loss_reconstructed
                + self.decoder_latent_loss_weight * diff_reconstructed
            )

            loss_components["decoder_recon_loss_reconstructed"] = (
                recon_loss_reconstructed
            )
            loss_components["emb_decoder_recon_loss_reconstructed"] = emb_recon_loss_reconstructed
            loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
            loss_components["decoder_loss_reconstructed"] = (
                decoder_loss_reconstructed
            )
            loss = loss + decoder_loss_reconstructed
            if hasattr(self.post_concat_projection, 'kl_loss'):
                kl = self.post_concat_projection.kl_loss  # scalar mean KL
                loss = loss + 1 * kl
                loss_components["projector_kl_loss"] = kl
        else:
            visual_reconstructed = None
            
        # Add visualization data to loss_components for three-way reconstruction comparison
        loss_components["visual_tgt"] = visual_tgt
        loss_components["visual_pred"] = visual_pred  
        loss_components["visual_reconstructed"] = visual_reconstructed
        loss_components["loss"] = loss
        return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        act_emb = self.encode_act(act)
        if self.concat_dim == 0:
            z[:, :, -1, :] = act_emb
        elif self.concat_dim == 1:
            act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
            act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
            z[..., -self.action_dim:] = act_repeated
        return z


    def rollout_original(self, obs_0, act):
        """
        Original rollout for concatenated (not mixed) features.
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z, _ = self.encode(obs_0, act_0)
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -inc:, ...]
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_obses = self.separate_emb(z)
        return z_obses, z

    def rollout(self, obs_0, act):
        """
        Rollout following original DINO WM strategy with 80D features and action replacement.
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                z_obses: dict with full 80D features for compatibility
                z: (b, t+n+1, num_patches, emb_dim) - 80D features [64D projected + 16D action]
        """
        num_obs_init = obs_0['visual'].shape[1]
        act_0 = act[:, :num_obs_init]
        action = act[:, num_obs_init:] 
        z, _ = self.encode(obs_0, act_0)  # Returns 80D features [64D projected + 16D action]
        t = 0
        inc = 1
        while t < action.shape[1]:
            z_pred = self.predict(z[:, -self.num_hist :])  # 80D → 80D prediction
            z_new = z_pred[:, -inc:, ...]
            # Following original DINO WM: replace action portion with planned actions
            z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
            z = torch.cat([z, z_new], dim=1)
            t += inc

        z_pred = self.predict(z[:, -self.num_hist :])
        z_new = z_pred[:, -1 :, ...] # take only the next pred
        z = torch.cat([z, z_new], dim=1)
        z_dct = self.separate_s_a(z)

        z_emb = self.emb_decoder(z_dct['projected'])
        z_obses = self.separate_emb(z_emb)
        
        return {'projected': z_dct['projected']}, z
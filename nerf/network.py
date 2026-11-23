import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetworkCond(NeRFRenderer):
    def __init__(
        self,
        encoding="hashgrid",
        encoding_dir="sphere_harmonics",
        encoding_bg="hashgrid",
        encoding_time1="frequency",
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        num_layers_bg=2,
        hidden_dim_bg=64,
        bound=1,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.basis = 11 + 1  # 16 #11 #8 #16 #24 #24 #16
        self.level = 16
        self.feat = 4
        self.total = self.feat * self.basis
        encodern = []
        for idxx in range(0, self.basis):
            encoder, self.in_dim3 = get_encoder(
                encoding,
                input_dim=3,
                num_levels=self.level,
                level_dim=self.feat,
                log2_hashmap_size=17,
                desired_resolution=2048 * bound,
            )
            encodern.append(encoder)
        self.encoder = nn.ModuleList(encodern)
        self.encoder4, self.in_dim4 = get_encoder("frequency", input_dim=4, multires=10)
        self.encoder2, self.in_dim2 = get_encoder(
            encoding, num_levels=self.level, level_dim=self.feat, log2_hashmap_size=17, desired_resolution=2048 * bound
        )
        ##########################################################################
        self.num_layers_deform = 4
        self.hidden_dim_deform = 128

        self.num_layers_hypern = 4
        self.hidden_dim_hypern = 64
        ####################################################################

        self.color_codexx = torch.nn.Parameter(torch.zeros(10000, 4).cuda())
        self.latentxx = torch.nn.Parameter(torch.zeros(10000, self.basis).unsqueeze(-1).unsqueeze(-1).cuda())
        self.s_max = torch.nn.Softmax(dim=0)

        self.l000 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l0 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l0[0] = 1
        self.l1 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l1[1] = 1
        self.l2 = torch.zeros_like(self.latentxx[20]).float().cuda().detach()
        self.l2[2] = 1

        self.l3 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l3[3] = 1
        self.l4 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l4[4] = 1
        self.l5 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l5[5] = 1
        self.l6 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l6[6] = 1
        self.l7 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l7[7] = 1
        self.l8 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l8[8] = 1
        self.l9 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l9[9] = 1
        self.l10 = torch.zeros_like(self.latentxx[0]).float().cuda().detach()
        self.l10[10] = 1

        deform_net = []
        for l in range(self.num_layers_deform):
            if l == 0:
                in_dim = self.in_dim3
            else:
                in_dim = self.hidden_dim_deform

            if l == self.num_layers_deform - 1:
                out_dim = 3
            else:
                out_dim = self.hidden_dim_deform

            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim2
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(
                encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=18, desired_resolution=1024
            )  # much smaller hashgrid

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d, t1, iddd, epoch):

        # sigma

        x1 = torch.stack([enc(x, bound=self.bound) for enc in self.encoder])

        x2 = self.encoder2(x, bound=self.bound)
        x3 = ((epoch[0, 0, 0 : self.basis, :, :]) * (x1)).sum(dim=0) + x2

        h = x3
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x, t1, iddd, epoch, fr=None, threshold=160, condi=None):
        if iddd >= threshold:
            for param in self.sigma_net.parameters():
                param.requires_grad = True
            for param in self.color_net.parameters():
                param.requires_grad = True
            for param in self.encoder_dir.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.encoder2.parameters():
                param.requires_grad = True
            for param in self.encoder_bg.parameters():
                param.requires_grad = True
            for param in self.bg_net.parameters():
                param.requires_grad = True
            for param in self.deform_net.parameters():
                param.requires_grad = False
            # for param in self.encoder4.parameters():
            # for param in self.hypern_net.parameters():
        else:
            self.sigma_net.requires_grad = True
            self.color_net.requires_grad = True
            self.encoder.requires_grad = True
            self.encoder_dir.requires_grad = True
            self.encoder2.requires_grad = True
            self.encoder_bg.requires_grad = True
            self.bg_net.requires_grad = True
            self.encoder4.requires_grad = True
            if iddd > 39:
                for param in self.deform_net.parameters():
                    param.requires_grad = False
                self.latentxx.requires_grad = False
                # for param in self.hypern_net.parameters():
                # for param in self.encoder.parameters():
            else:
                self.deform_net.requires_grad = True
                self.latentxx.requires_grad = True
        latent3 = None
        if t1 == 0:
            latent2 = self.latentxx[0]
            jj = [-1]
        elif (t1 >= fr[0][0]) and t1 <= (fr[0][1]):
            latent2 = self.latentxx[fr[0][0]]
            jj = [-1, 0]
        elif (t1 >= fr[1][0]) and t1 <= (fr[1][1]):
            latent2 = self.latentxx[fr[1][0]]
            jj = [-1, 1]
        elif (t1 >= fr[2][0]) and t1 <= (fr[2][1]):
            latent2 = self.latentxx[fr[2][0]]
            jj = [-1, 2]
        elif (t1 >= fr[3][0]) and t1 <= (fr[3][1]):
            latent2 = self.latentxx[fr[3][0]]
            jj = [-1, 3]
        elif (t1 >= fr[4][0]) and t1 <= (fr[4][1]):
            latent2 = self.latentxx[fr[4][0]]
            jj = [-1, 4]
        elif (t1 >= fr[5][0]) and t1 <= (fr[5][1]):
            latent2 = self.latentxx[fr[5][0]]
            jj = [-1, 5]
        elif (t1 >= fr[6][0]) and t1 <= (fr[6][1]):
            latent2 = self.latentxx[fr[6][0]]
            jj = [-1, 6]
        elif (t1 >= fr[7][0]) and t1 <= (fr[7][1]):
            latent2 = self.latentxx[fr[7][0]]
            jj = [-1, 7]
        elif (t1 >= fr[8][0]) and t1 <= (fr[8][1]):
            latent2 = self.latentxx[fr[8][0]]
            jj = [-1, 8]
        elif (t1 >= fr[9][0]) and t1 <= (fr[9][1]):
            latent2 = self.latentxx[fr[9][0]]
            jj = [-1, 9]
        elif (t1 >= fr[10][0]) and t1 <= (fr[10][1]):
            latent2 = self.latentxx[fr[10][0]]
            jj = [-1, 10]
        else:
            if condi is None:
                latent2 = (self.latentxx[t1] + self.latentxx[t1 - 1] + self.latentxx[t1 + 1]) / 3

        if not self.training:  # False:
            dx = []
            for enc in self.encoder:
                h2 = enc(x, bound=self.bound)
                for l in range(self.num_layers_deform):
                    h2 = self.deform_net[l](h2)
                    if l != self.num_layers_deform - 1:
                        h2 = F.relu(h2, inplace=True)
                dx.append(h2)
            defor = torch.stack(dx)

            if t1 != 0:

                if condi is None:
                    x4 = ((latent2[0 : self.basis - 1]) * (epoch) * (defor[0 : self.basis - 1])).sum(dim=0) + (
                        (~(epoch.bool())).float()
                    ) * defor[self.basis - 1 : self.basis]
                else:
                    x4 = ((condi[0].unsqueeze(-1).unsqueeze(-1)) * (epoch) * (defor[0 : self.basis - 1])).sum(dim=0) + (
                        (~(epoch.bool())).float()
                    ) * defor[self.basis - 1 : self.basis]
            else:
                x4 = 1 * defor[self.basis - 1 : self.basis]
        else:
            if t1 != 0:
                # if iddd>=threshold:
                dx = []
                for enc in self.encoder:
                    h2 = enc(x, bound=self.bound)
                    for l in range(self.num_layers_deform):
                        h2 = self.deform_net[l](h2)
                        if l != self.num_layers_deform - 1:
                            h2 = F.relu(h2, inplace=True)
                    dx.append(h2)
                defor = torch.stack(dx)
                x4 = ((latent2[0 : self.basis - 1]) * (epoch) * (defor[0 : self.basis - 1])).sum(dim=0) + (
                    (~(epoch.bool())).float()
                ) * defor[self.basis - 1 : self.basis]

            else:
                dx = []
                for kk in jj:
                    h2 = (self.encoder[kk])(x, bound=self.bound)
                    for l in range(self.num_layers_deform):
                        h2 = self.deform_net[l](h2)
                        if l != self.num_layers_deform - 1:
                            h2 = F.relu(h2, inplace=True)
                    dx.append(h2)
                defor = torch.stack(dx)
                x4 = (1) * defor

        #     if iddd>10:
        h = self.encoder2(x + (x4), bound=self.bound)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        loss = 0
        return (
            {
                "sigma": sigma,
                "geo_feat": geo_feat,
            },
            x4,
            loss,
            None,
        )

    def background(self, x, d):

        h = self.encoder_bg(x)  # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, color_code=None, geo_feat=None, **kwargs):
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]
            if color_code is not None:
                color_code = color_code[mask]

        d = self.encoder_dir(d)
        if color_code is not None:
            h = torch.cat([d, geo_feat, color_code], dim=-1)
        else:
            h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)
        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)
        else:
            rgbs = h

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {"params": self.encoder.parameters(), "lr": lr},
            {"params": self.sigma_net.parameters(), "lr": lr},
            {"params": self.encoder_dir.parameters(), "lr": lr},
            {"params": self.color_net.parameters(), "lr": lr},
            {"params": self.encoder2.parameters(), "lr": lr},
            {"params": self.deform_net.parameters(), "lr": 1e-3},
            {"params": self.latentxx, "lr": 0.5e-3},
            {"params": self.color_codexx, "lr": 0.5e-3},
            {"params": self.encoder4.parameters(), "lr": lr},
        ]
        if self.bg_radius > 0:
            params.append({"params": self.encoder_bg.parameters(), "lr": lr})
            params.append({"params": self.bg_net.parameters(), "lr": lr})

        return params


class MappingNetwork(NeRFRenderer):
    def __init__(
        self,
        encoding="tiled",
        encoding_dir="sphere_harmonics",
        encoding_bg="hashgrid",
        encoding_time1="frequency",
        num_layers=2,
        hidden_dim=64,
        geo_feat_dim=15,
        num_layers_color=3,
        hidden_dim_color=64,
        num_layers_bg=2,
        hidden_dim_bg=64,
        bound=1,
        **kwargs,
    ):
        super().__init__(bound, **kwargs)

        self.basis = 16
        self.num_layers_deform = 4
        self.hidden_dim_deform = 128
        self.encoder_deform, self.in_dim_deform = get_encoder("frequency", input_dim=self.basis, multires=10)

        deform_net = []
        for l in range(self.num_layers_deform):
            if l == 0:
                in_dim = self.in_dim_deform
            else:
                in_dim = self.hidden_dim_deform

            if l == self.num_layers_deform - 1:
                out_dim = 11
            else:
                out_dim = self.hidden_dim_deform

            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)

    def forward(self, epoch):

        enc_ori_x = self.encoder_deform(
            epoch[0, :, 0 : self.basis, 0, 0], bound=self.bound
        )  # [N, C] #self.encoder_deform(epoch[0,:,0:self.basis,0,0], bound=self.bound) # [N, C]
        deform = enc_ori_x
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)

        return deform

    def get_params(self, lr):

        params = [
            {"params": self.encoder_deform.parameters(), "lr": 1e-3},
            {"params": self.deform_net.parameters(), "lr": 1e-3},
        ]

        return params

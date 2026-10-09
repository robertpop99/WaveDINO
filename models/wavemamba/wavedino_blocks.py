# Terrain modulation and phase ramp head blocks of WaveDINO.
# Licensed under the Apache License 2.0 (see LICENSE in the repository root). These blocks are not derived from
# Wave-Mamba; the network that uses them (wavemamba.py) is under CC BY-NC-SA 4.0.
import torch
import torch.nn as nn
import torch.nn.functional as F


class TerrainConditioning(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # DEM + gradients
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, dim * 2, 3, padding=1)
        )

    def forward(self, x, dem):
        terrain = self.encoder(dem)

        gamma, beta = terrain.chunk(2, dim=1)

        return gamma * x + beta


class TerrainConditioningV2(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # DEM + gradients
            nn.ReLU(),
            nn.Conv2d(32, dim * 2, 3, padding=1)
        )

        self.gate = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x, dem):
        gamma_beta = self.encoder(dem)
        gamma, beta = gamma_beta.chunk(2, dim=1)

        terrain_mod = gamma * x + beta

        g = torch.sigmoid(self.gate(dem))

        return x + g * terrain_mod


class CoherenceFiLM(nn.Module):
    """
    Coherence-conditioned FiLM modulation:
        x' = x * (1 + gamma(coh)) + beta(coh)
    Optionally gated with a learnable scalar in [0,1] per channel.

    Args:
        channels: number of channels in x
        hidden: hidden channels for the coherence encoder
        use_gate: whether to use learnable channel-wise gating
        clamp_coh: whether to clamp coherence to [0,1]
    """
    def __init__(self, channels: int, hidden: int = 32, use_gate: bool = True, clamp_coh: bool = True):
        super().__init__()
        self.channels = channels
        self.clamp_coh = clamp_coh
        self.use_gate = use_gate

        # small CNN to map coherence -> (gamma, beta) fields
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * channels, kernel_size=1, padding=0),
        )

        # learnable gate per channel (initialized near zero -> starts as "do nothing")
        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))
        else:
            self.register_parameter("gate", None)

        # initialize last layer small to avoid destabilizing early training
        last = self.net[-1]
        if isinstance(last, nn.Conv2d):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, coh: torch.Tensor) -> torch.Tensor:
        """
        x:   [B, C, H, W]
        coh: [B, 1, Hc, Wc] or [B, Hc, Wc]
        """
        if coh.dim() == 3:
            coh = coh.unsqueeze(1)  # [B,1,H,W]

        # clamp coherence to [0,1] if desired
        if self.clamp_coh:
            coh = coh.clamp(0.0, 1.0)

        # resize coherence to match feature resolution
        if coh.shape[-2:] != x.shape[-2:]:
            coh = F.interpolate(coh, size=x.shape[-2:], mode="bilinear", align_corners=False)

        gb = self.net(coh)  # [B, 2C, H, W]
        gamma, beta = gb.chunk(2, dim=1)  # each [B, C, H, W]

        # optional channel-wise gate (sigmoid to [0,1])
        if self.use_gate:
            g = torch.sigmoid(self.gate)  # [1,C,1,1]
            gamma = gamma * g
            beta = beta * g

        # FiLM (1 + gamma) so the identity mapping is easy
        return x * (1.0 + gamma) + beta


class DWSeparableConv(nn.Module):
    """Depthwise separable conv with GroupNorm for efficiency and stability."""
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, d=1, groups=8):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, k, s, p, dilation=d, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.gn = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        return self.act(x)


class TerrainConditioningV3(nn.Module):
    """
    Improvements over V2:
      - DEM feature encoder with multi-scale (dilation) and normalization
      - Gate derived from learned DEM features (not raw DEM)
      - Zero-init last conv to start at identity, plus learnable residual scale alpha
      - Optional cross gating with x (toggleable)
    """
    def __init__(self, dim, dem_in=3, groups=8, cross_gate=False):
        super().__init__()

        # 1) DEM encoder (multi-scale context via dilation)
        self.dem_enc = nn.Sequential(
            nn.Conv2d(dem_in, 32, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, 32), 32),
            nn.SiLU(inplace=True),

            DWSeparableConv(32, 64, k=3, p=1, d=1, groups=groups),
            DWSeparableConv(64, 64, k=3, p=2, d=2, groups=groups),   # larger RF
            nn.Conv2d(64, 2 * dim, 1, bias=True)                     # -> [γ_raw, β]
        )

        # 2) Gate from learned DEM features (or cross-gated with x)
        self.cross_gate = cross_gate
        gate_in = (2 * dim) if not cross_gate else (2 * dim + dim)
        self.gate = nn.Sequential(
            nn.Conv2d(gate_in, dim // 2, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 1, bias=True)
        )

        # 3) Residual scaling; start near identity
        self.alpha = nn.Parameter(torch.zeros(1))

        # 4) Zero-init the last layer of dem_enc so γ, β start at ~0
        nn.init.zeros_(self.dem_enc[-1].weight)
        if self.dem_enc[-1].bias is not None:
            nn.init.zeros_(self.dem_enc[-1].bias)

        # 5) Optionally, lightly scale initial gate bias negative to favor identity
        gate_last = self.gate[-1]
        if gate_last.bias is not None:
            nn.init.constant_(gate_last.bias, -2.0)   # sigmoid(-2) ≈ 0.12

    def forward(self, x, dem, dem_mask=None):
        """
        x:   [B, C=dim, H, W]
        dem: [B, dem_in, H, W]  (DEM + gradients already normalized)
        dem_mask: optional [B,1,H,W] for voids; 1=valid, 0=void
        """
        # Encode DEM into FiLM parameters
        film = self.dem_enc(dem)                  # [B, 2*dim, H, W]
        gamma, beta = film.chunk(2, dim=1)        # [B, dim, H, W] each

        # Optional mask to avoid leaking void regions
        if dem_mask is not None:
            gamma = gamma * dem_mask
            beta  = beta  * dem_mask

        # Compute gate from learned features (optional cross gating with x)
        gate_input = film if not self.cross_gate else torch.cat([film, x], dim=1)
        g = torch.sigmoid(self.gate(gate_input))  # [B,1,H,W]

        # FiLM + gated residual with learnable scale
        terrain_mod = gamma * x + beta
        y = x + self.alpha * g * terrain_mod
        return y


class PhaseRampHeadZernike(nn.Module):
    def __init__(self, in_channels, max_ramp=0.1, max_coeff=0.3):
        super().__init__()

        # Maximum total ramp magnitude
        self.max_ramp = max_ramp

        # Maximum coefficient for a, b, c before scaling
        self.max_coeff = max_coeff

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Predict Zernike basis coefficients (c0, c1, c2, c3, c4, c5)
        self.poly_head = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # c0, c1, c2, c3, c4, c5
        )

        # Predict global ramp scale (now symmetric)
        self.scale_head = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, x, coord_grid=None):
        """
        x: (B, C, H, W)
        coord_grid: (B, 2, H, W) with values in [-1, 1]
                    channel 0 = x coords, channel 1 = y coords
        """

        B, C, H, W = x.shape

        # Global pooling
        pooled = self.pool(x).view(B, C)

        # Predict parameters
        coeffs = self.poly_head(pooled)  # (B, 6)
        scale = self.scale_head(pooled)  # (B, 1)

        # Stabilize coefficients
        coeffs = torch.tanh(coeffs) * self.max_coeff

        c0, c1, c2, c3, c4, c5 = torch.chunk(coeffs, 6, dim=1)

        c0 = c0.view(B, 1, 1, 1)
        c1 = c1.view(B, 1, 1, 1)
        c2 = c2.view(B, 1, 1, 1)
        c3 = c3.view(B, 1, 1, 1)
        c4 = c4.view(B, 1, 1, 1)
        c5 = c5.view(B, 1, 1, 1)

        scale = scale.view(B, 1, 1, 1) * self.max_ramp

        # Ensure coord grid exists
        if coord_grid is None:
            # Create normalized coordinate grid [-1, 1]
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=x.device),
                torch.linspace(-1, 1, W, device=x.device),
                indexing='ij'
            )
            coord_grid = torch.stack([xx, yy], dim=0)  # (2, H, W)
            coord_grid = coord_grid.unsqueeze(0).repeat(B, 1, 1, 1)

        x_coord = coord_grid[:, 0:1, :, :]
        y_coord = coord_grid[:, 1:2, :, :]

        r2 = x_coord ** 2 + y_coord ** 2

        # Compute ramp
        ramp = (
                c0 +  # piston
                c1 * x_coord +  # tilt x
                c2 * y_coord +  # tilt y
                c3 * (2 * r2 - 1) +  # defocus
                c4 * (x_coord ** 2 - y_coord ** 2) +  # astigmatism
                c5 * (x_coord * y_coord)  # shear
        )

        # Apply scale
        ramp = ramp * scale

        return ramp


class PhaseRampHeadV5(nn.Module):
    def __init__(self, in_channels, max_ramp=0.1, max_coeff=0.3):
        super().__init__()

        # Maximum total ramp magnitude
        self.max_ramp = max_ramp

        # Maximum coefficient for a, b, c before scaling
        self.max_coeff = max_coeff

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Predict polynomial coefficients (a, b, c, d, e, f)
        self.poly_head = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # a, b, c, d, e, f
        )

        # Predict global ramp scale (now symmetric)
        self.scale_head = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, x, coord_grid=None):
        """
        x: (B, C, H, W)
        coord_grid: (B, 2, H, W) with values in [-1, 1]
                    channel 0 = x coords, channel 1 = y coords
        """

        B, C, H, W = x.shape

        # Global pooling
        pooled = self.pool(x).view(B, C)

        # Predict parameters
        coeffs = self.poly_head(pooled)  # (B, 6)
        scale = self.scale_head(pooled)  # (B, 1)

        # Stabilize coefficients
        coeffs = torch.tanh(coeffs) * self.max_coeff

        a, b, c, d, e, f = torch.chunk(coeffs, 6, dim=1)

        a = a.view(B, 1, 1, 1)
        b = b.view(B, 1, 1, 1)
        c = c.view(B, 1, 1, 1)
        d = d.view(B, 1, 1, 1)
        e = e.view(B, 1, 1, 1)
        f = f.view(B, 1, 1, 1)

        scale = scale.view(B, 1, 1, 1) * self.max_ramp

        # Ensure coord grid exists
        if coord_grid is None:
            # Create normalized coordinate grid [-1, 1]
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=x.device),
                torch.linspace(-1, 1, W, device=x.device),
                indexing='ij'
            )
            coord_grid = torch.stack([xx, yy], dim=0)  # (2, H, W)
            coord_grid = coord_grid.unsqueeze(0).repeat(B, 1, 1, 1)

        x_coord = coord_grid[:, 0:1, :, :]
        y_coord = coord_grid[:, 1:2, :, :]

        # Compute ramp
        ramp = (
                a * x_coord +
                b * y_coord +
                c +
                d * (x_coord ** 2 - 1/3) + # -1/3 to keep it zero-mean over [-1,1]
                e * (y_coord ** 2 - 1/3) +
                f * x_coord * y_coord
        )

        # Apply scale
        ramp = ramp * scale

        return ramp


class PhaseRampHeadV4(nn.Module):
    def __init__(self, in_channels, max_ramp=0.1, max_coeff=0.3):
        super().__init__()

        # Maximum total ramp magnitude
        self.max_ramp = max_ramp

        # Maximum coefficient for a, b, c before scaling
        self.max_coeff = max_coeff

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Predict slope (a, b)
        self.ab_head = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # a, b
        )

        # Predict offset (c)
        self.c_head = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # c
        )

        # Predict global ramp scale (now symmetric)
        self.scale_head = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # [-1, 1]
        )

    def forward(self, x, coord_grid=None):
        """
        x: (B, C, H, W)
        coord_grid: (B, 2, H, W) with values in [-1, 1]
                    channel 0 = x coords, channel 1 = y coords
        """

        B, C, H, W = x.shape

        # Global pooling
        pooled = self.pool(x).view(B, C)

        # Predict parameters
        ab = self.ab_head(pooled)              # (B, 2)
        c = self.c_head(pooled)                # (B, 1)
        scale = self.scale_head(pooled)        # (B, 1)

        # Stabilize coefficients
        ab = torch.tanh(ab) * self.max_coeff
        c = torch.tanh(c) * self.max_coeff

        a = ab[:, 0].view(B, 1, 1, 1)
        b = ab[:, 1].view(B, 1, 1, 1)
        c = c.view(B, 1, 1, 1)

        scale = scale.view(B, 1, 1, 1) * self.max_ramp

        # Ensure coord grid exists
        if coord_grid is None:
            # Create normalized coordinate grid [-1, 1]
            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=x.device),
                torch.linspace(-1, 1, W, device=x.device),
                indexing='ij'
            )
            coord_grid = torch.stack([xx, yy], dim=0)  # (2, H, W)
            coord_grid = coord_grid.unsqueeze(0).repeat(B, 1, 1, 1)

        x_coord = coord_grid[:, 0:1, :, :]
        y_coord = coord_grid[:, 1:2, :, :]

        # Compute ramp
        ramp = a * x_coord + b * y_coord + c

        # Apply scale
        ramp = ramp * scale

        return ramp


class PhaseRampHeadV3(nn.Module):

    def __init__(self, in_channels, max_ramp=0.1):
        super().__init__()

        # max ramp in rad after normalisation (we use s=100 for normalisation)
        # in this case 10 rad
        self.max_ramp = max_ramp

        self.pool = nn.AdaptiveAvgPool2d(1)

        # ramp parameter prediction
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # a,b,c
        )

        # adaptive ramp magnitude
        self.scale_head = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        B, C, H, W = x.shape

        feat = self.pool(x).flatten(1)

        params = self.fc(feat)
        scale = self.scale_head(feat) * self.max_ramp  # (B,1)

        a, b, c = params.T

        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,H,device=x.device),
            torch.linspace(-1,1,W,device=x.device),
            indexing="ij"
        )

        ramp = (
            a[:,None,None]*xx +
            b[:,None,None]*yy +
            c[:,None,None]
        )

        ramp = ramp.unsqueeze(1)  # B x 1 x H x W

        # adaptive amplitude constraint
        ramp = scale[:, :, None, None] * torch.tanh(ramp)

        return ramp


class PhaseRampHeadV2(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 6)  # a,b,c,d,e,f
        )

    def forward(self, x):

        B, C, H, W = x.shape

        feat = self.pool(x).flatten(1)
        params = self.fc(feat)

        a,b,c,d,e,f = params.T

        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,H,device=x.device),
            torch.linspace(-1,1,W,device=x.device),
            indexing="ij"
        )

        ramp = (
            a[:,None,None]*xx +
            b[:,None,None]*yy +
            c[:,None,None] +
            d[:,None,None]*(xx**2) +
            e[:,None,None]*(yy**2) +
            f[:,None,None]*(xx*yy)
        )

        return ramp.unsqueeze(1)


class PhaseRampHead(nn.Module):

    def __init__(self, in_channels):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3)   # a,b,c
        )

    def forward(self, x):

        B, C, H, W = x.shape

        feat = self.pool(x).flatten(1)
        params = self.fc(feat)

        a = params[:,0]
        b = params[:,1]
        c = params[:,2]

        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,H,device=x.device),
            torch.linspace(-1,1,W,device=x.device),
            indexing="ij"
        )

        ramp = a[:,None,None]*xx + b[:,None,None]*yy + c[:,None,None]

        return ramp.unsqueeze(1)

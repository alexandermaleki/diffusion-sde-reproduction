from types import SimpleNamespace


def create_ddpmpp_vp_config(image_size=32, image_channels=3):
    config = SimpleNamespace()
    config.training = SimpleNamespace()
    config.training.continuous = True

    config.data = SimpleNamespace()
    config.data.image_size = image_size
    config.data.num_channels = image_channels
    config.data.centered = True

    config.model = SimpleNamespace()
    config.model.name = "ncsnpp"
    config.model.sigma_min = 0.01
    config.model.sigma_max = 50
    config.model.num_scales = 1000
    config.model.beta_min = 0.1
    config.model.beta_max = 20.0

    config.model.nf = 64
    config.model.ch_mult = (1, 2, 2, 2)
    config.model.num_res_blocks = 2
    config.model.attn_resolutions = (16,)
    config.model.dropout = 0.1

    config.model.scale_by_sigma = False
    config.model.normalization = "GroupNorm"
    config.model.nonlinearity = "swish"
    config.model.resamp_with_conv = True
    config.model.conditional = True
    config.model.fir = True
    config.model.fir_kernel = [1, 3, 3, 1]
    config.model.skip_rescale = True
    config.model.resblock_type = "biggan"
    config.model.progressive = "none"
    config.model.progressive_input = "residual"
    config.model.progressive_combine = "sum"
    config.model.attention_type = "ddpm"
    config.model.init_scale = 0.0
    config.model.embedding_type = "positional"
    config.model.fourier_scale = 16
    config.model.conv_size = 3

    return config


def create_ncsnpp_ve_config(image_size=32, image_channels=3):
    config = SimpleNamespace()
    config.training = SimpleNamespace()
    config.training.sde = "vesde"
    config.training.continuous = True

    config.data = SimpleNamespace()
    config.data.image_size = image_size
    config.data.num_channels = image_channels
    config.data.centered = True

    config.model = SimpleNamespace()
    config.model.name = "ncsnpp"
    config.model.sigma_min = 0.01
    config.model.sigma_max = 50.0
    config.model.num_scales = 1000
    config.model.scale_by_sigma = True
    config.model.ema_rate = 0.999
    config.model.normalization = "GroupNorm"
    config.model.nonlinearity = "swish"
    config.model.nf = 128
    config.model.ch_mult = (1, 2, 2, 2)
    config.model.num_res_blocks = 4
    config.model.attn_resolutions = (16,)
    config.model.dropout = 0.1
    config.model.resamp_with_conv = True
    config.model.conditional = True
    config.model.fir = True
    config.model.fir_kernel = [1, 3, 3, 1]
    config.model.skip_rescale = True
    config.model.resblock_type = "biggan"
    config.model.progressive = "none"
    config.model.progressive_input = "residual"
    config.model.progressive_combine = "sum"
    config.model.attention_type = "ddpm"
    config.model.init_scale = 0.0
    config.model.embedding_type = "fourier"
    config.model.fourier_scale = 16
    config.model.conv_size = 3

    return config

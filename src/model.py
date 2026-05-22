from diffusers import UNet2DModel

def get_satellite_unet():
    """
    Initializes a 2D U-Net tailored for 2-channel satellite data.
    """
    model = UNet2DModel(
        # Base resolution of the maps
        # From the documentation: Dimensions must be a multiple of 2 ** (len(block_out_channels) - 1)
        # So, Dimensions must be a multiple of 2^(3) = 8
        sample_size=(32, 32),

        # Number of input channels: LST and Precipitation --> 2
        in_channels=2,

        # Number of output channels: Predicted noise for LST and Precipitation --> 2
        out_channels=2,

        # This dictates the feature depth after each filter (number of channels)
        # 4 blocks means we downsample 3 times (divisible by 8 constraint)
        block_out_channels=(64, 128, 256, 512),

        # Standard Downsampling blocks
        down_block_types=(
            "DownBlock2D",      # 64 channels (32x32)
            "DownBlock2D",      # 128 channels (16x16)
            "AttnDownBlock2D",  # 256 channels (8x8)
            "AttnDownBlock2D",  # 512 channels (4x4)
        ),

        # Standard Upsampling blocks
        up_block_types=(
            "AttnUpBlock2D",    # 512 channels (4x4)
            "AttnUpBlock2D",    # 256 channels (8x8)
            "UpBlock2D",        # 128 channels (16x16)
            "UpBlock2D",        # 64 channels (32x32)
        ),
    )

    return model


# Quick test to make sure it compiles
if __name__ == "__main__":
    test_model = get_satellite_unet()
    print(f"U-Net successfully initialized with {sum(p.numel() for p in test_model.parameters()):,} parameters.")
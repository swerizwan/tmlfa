from backbones import get_model
from analysis import subnets

def build_model(cfg):
    """
    Builds and returns a model composed of a backbone network and additional modules.

    Args:
        cfg (object): Configuration object containing model parameters and settings.

    Returns:
        subnets.ModelBox: The constructed model.
    """

    # Get the backbone network based on the configuration
    backbone = get_model(cfg.network, num_features=cfg.embedding_size)
    """
    The backbone network is the primary feature extractor.
    It is obtained using the get_model function from the backbones module.
    The type of network and the number of embedding features are specified in the configuration.
    """

    # Initialize the Feature Attention Module (FAM)
    fam = subnets.FeatureAttentionModule(
        in_chans=cfg.fam_in_chans,  # Number of input channels
        kernel_size=cfg.fam_kernel_size,  # Kernel size for convolutional layers
        conv_shared=cfg.fam_conv_shared,  # Whether to share convolutional layers
        conv_mode=cfg.fam_conv_mode,  # Mode of convolution (e.g., 'same', 'valid')
        channel_attention=cfg.fam_channel_attention,  # Whether to use channel attention
        spatial_attention=cfg.fam_spatial_attention,  # Whether to use spatial attention
        pooling=cfg.fam_pooling,  # Type of pooling to use (e.g., 'max', 'avg')
        la_num_list=cfg.fam_la_num_list  # List of numbers for local attention
    )
    """
    The Feature Attention Module (FAM) is designed to enhance feature extraction.
    It uses various techniques such as channel and spatial attention to improve feature quality.
    The specific configurations for FAM are provided in the cfg object.
    """

    # Initialize the Task Specific Subnets (TSS)
    tss = subnets.TaskSpecificSubnets()
    """
    Task Specific Subnets (TSS) are designed to handle task-specific processing.
    These subnets can be customized to handle different tasks or domains.
    """

    # Initialize the Result Module (OM)
    om = subnets.resultModule()
    """
    The Result Module (OM) is responsible for generating the final output.
    It processes the features from the previous modules and produces the model's predictions.
    """

    # Combine all components into the ModelBox
    model = subnets.ModelBox(
        backbone=backbone,  # The backbone network
        fam=fam,  # The Feature Attention Module
        tss=tss,  # The Task Specific Subnets
        om=om,  # The Result Module
        feature=cfg.fam_feature  # Feature configuration for FAM
    )
    """
    The ModelBox is a container that integrates all the components.
    It ensures that the data flows correctly through the backbone, FAM, TSS, and OM.
    The feature configuration for FAM is also passed to the ModelBox.
    """

    return model
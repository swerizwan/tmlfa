def get_model(name, **kwargs):
    """
    Retrieve a model based on the provided name and keyword arguments.

    Args:
        name (str): Name of the model to retrieve.
        **kwargs: Additional keyword arguments for model initialization.

    Returns:
        nn.Module: The initialized model.

    Raises:
        ValueError: If the model name is not recognized.
    """
    if name == "swin_t":
        # Retrieve the number of features from the keyword arguments, defaulting to 512 if not provided
        num_features = kwargs.get("num_features", 512)
        
        # Import the SwinTransformer model from the swin module
        from .swin import SwinTransformer
        
        # Initialize and return the SwinTransformer model with the specified number of classes
        return SwinTransformer(num_classes=num_features)

    else:
        # Raise a ValueError if the model name is not recognized
        raise ValueError("Model name not recognized")

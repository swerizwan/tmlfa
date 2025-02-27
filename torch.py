import numpy as np
import onnx
import torch


def convert_onnx(net, path_module, result, opset=11, simplify=False):
    """
    Convert a PyTorch model to ONNX format.

    Args:
        net (torch.nn.Module): The PyTorch model to be converted.
        path_module (str): Path to the PyTorch model weights.
        result (str): Path to save the resulting ONNX model.
        opset (int, optional): ONNX opset version. Defaults to 11.
        simplify (bool, optional): Whether to simplify the ONNX model. Defaults to False.
    """
    # Ensure the input model is a PyTorch nn.Module
    assert isinstance(net, torch.nn.Module)

    # Create a random input image for model tracing
    img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
    img = img.astype(np.float)
    img = (img / 255. - 0.5) / 0.5  # Normalize in torch style
    img = img.transpose((2, 0, 1))  # Change to channel-first format
    img = torch.from_numpy(img).unsqueeze(0).float()  # Add batch dimension and convert to tensor

    # Load the model weights
    weight = torch.load(path_module)
    net.load_state_dict(weight, strict=True)
    net.eval()  # Set the model to evaluation mode

    # Export the model to ONNX format
    torch.onnx.export(net, img, result, input_names=["data"], keep_initializers_as_inputs=False, verbose=False, opset_version=opset)

    # Load the exported ONNX model
    model = onnx.load(result)
    graph = model.graph

    # Modify the input shape to be dynamic
    graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'

    # Simplify the ONNX model if required
    if simplify:
        from onnxsim import simplify
        model, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"

    # Save the ONNX model
    onnx.save(model, result)


if __name__ == '__main__':
    import os
    import argparse
    from backbones import get_model

    # Define the command-line argument parser
    parser = argparse.ArgumentParser(description='ArcFace PyTorch to onnx')
    parser.add_argument('input', type=str, help='input backbone.pth file or path')
    parser.add_argument('--result', type=str, default=None, help='result onnx path')
    parser.add_argument('--network', type=str, default=None, help='backbone network')
    parser.add_argument('--simplify', type=bool, default=False, help='onnx simplify')
    args = parser.parse_args()

    # Determine the input file path
    input_file = args.input
    if os.path.isdir(input_file):
        input_file = os.path.join(input_file, "model.pt")
    assert os.path.exists(input_file), "Input file does not exist"

    # Ensure the network type is specified
    assert args.network is not None, "Network type must be specified"
    print(args)

    # Get the backbone model
    backbone_onnx = get_model(args.network, dropout=0.0, fp16=False, num_features=512)

    # Determine the result file path
    if args.result is None:
        args.result = os.path.join(os.path.dirname(args.input), "model.onnx")

    # Convert the model to ONNX format
    convert_onnx(backbone_onnx, input_file, args.result, simplify=args.simplify)
import torch
import torch.nn as nn

class AgeLoss(nn.Module):
    """
    Custom loss function for age estimation tasks.

    This loss function combines a mean squared error (MSE) term with a term that
    penalizes large differences more heavily as training progresses. The idea is
    to start with a smooth loss landscape and gradually increase the penalty for
    larger errors as the model learns.

    Args:
        total_iter (int): Total number of training iterations.
        sigma (float, optional): Standard deviation for the exponential term. Defaults to 3.0.
    """
    def __init__(self, total_iter, sigma=3.0):
        super().__init__()

        self.sigma = sigma  # Standard deviation for the exponential term
        self.total_iter = total_iter  # Total number of training iterations

    def forward(self, result, label, current_iter):
        """
        Compute the loss for the given predictions and labels.

        Args:
            result (Tensor): Predicted ages.
            label (Tensor): Ground truth ages.
            current_iter (int): Current training iteration.

        Returns:
            Tensor: Computed loss.
        """
        result = result.to(torch.float32)  # Ensure the result tensor is of type float32
        label = label.to(torch.float32)  # Ensure the label tensor is of type float32

        # Compute the learning annealing factor (la)
        la = current_iter / self.total_iter

        # Compute the squared difference between predictions and labels
        dif_squ = (result - label) ** 2

        # Combine the MSE term and the exponential term
        loss = (1 - la) / 2 * dif_squ + la * (1 - torch.exp(-dif_squ / (2 * self.sigma ** 2)))

        # Compute the mean loss over the batch
        loss = torch.mean(loss)

        return loss  # Return the computed loss
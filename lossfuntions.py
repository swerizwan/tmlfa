import torch
import math


class CombinedMarginLoss(torch.nn.Module):
    """
    Combined Margin Loss module.

    This module implements a combined margin loss function, which can be used for face recognition tasks.
    It supports different margin types and interclass filtering.

    Args:
        s (float): Scaling factor.
        m1 (float): Margin parameter 1.
        m2 (float): Margin parameter 2.
        m3 (float): Margin parameter 3.
        interclass_filtering_threshold (float, optional): Threshold for interclass filtering. Defaults to 0.
    """
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        """
        Forward pass of the CombinedMarginLoss module.

        Args:
            logits (torch.Tensor): Logits from the model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Modified logits after applying the margin loss.
        """
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
            cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
            if self.easy_margin:
                final_target_logit = torch.where(
                    target_logit > 0, cos_theta_m, target_logit)
            else:
                final_target_logit = torch.where(
                    target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        
        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise ValueError("Unsupported margin configuration")

        return logits


class ArcFace(torch.nn.Module):
    """
    ArcFace loss module.

    This module implements the ArcFace loss function, which is a type of margin-based loss function.
    It is commonly used in face recognition tasks.

    Args:
        s (float, optional): Scaling factor. Defaults to 64.0.
        margin (float, optional): Margin parameter. Defaults to 0.5.
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.scale = s
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass of the ArcFace module.

        Args:
            logits (torch.Tensor): Logits from the model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Modified logits after applying the ArcFace loss.
        """
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * self.cos_m - sin_theta * self.sin_m  # cos(target+margin)
        if self.easy_margin:
            final_target_logit = torch.where(
                target_logit > 0, cos_theta_m, target_logit)
        else:
            final_target_logit = torch.where(
                target_logit > self.theta, cos_theta_m, target_logit - self.sinmm)

        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.scale
        return logits


class CosFace(torch.nn.Module):
    """
    CosFace loss module.

    This module implements the CosFace loss function, which is another type of margin-based loss function.
    It is commonly used in face recognition tasks.

    Args:
        s (float, optional): Scaling factor. Defaults to 64.0.
        m (float, optional): Margin parameter. Defaults to 0.40.
    """
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Forward pass of the CosFace module.

        Args:
            logits (torch.Tensor): Logits from the model.
            labels (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Modified logits after applying the CosFace loss.
        """
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits
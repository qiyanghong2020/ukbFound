import torch
import torch.nn.functional as F


def masked_mse_loss(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the masked MSE loss between input and target.
    Args:
        input (torch.Tensor): The predicted values.
        target (torch.Tensor): The ground truth values.
        mask (torch.Tensor): The mask tensor indicating the valid regions.

    Returns:
        torch.Tensor: The masked MSE loss.
    """
    # Ensure the mask is a float tensor
    mask = mask.float()

    # Compute the MSE loss only for the masked elements
    loss = F.mse_loss(input * mask, target * mask, reduction="sum")

    # Normalize the loss by the number of valid (non-zero) elements in the mask
    return loss / mask.sum()


def criterion_neg_log_bernoulli(
    input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Compute the negative log-likelihood of Bernoulli distribution
    """
    mask = mask.float()
    bernoulli = torch.distributions.Bernoulli(probs=input)
    masked_log_probs = bernoulli.log_prob((target > 0).float()) * mask
    return -masked_log_probs.sum() / mask.sum()


def masked_relative_error(
    input: torch.Tensor, target: torch.Tensor, mask: torch.LongTensor
) -> torch.Tensor:
    """
    Compute the masked relative error between input and target.
    """
    assert mask.any()
    loss = torch.abs(input[mask] - target[mask]) / (target[mask] + 1e-6)
    return loss.mean()

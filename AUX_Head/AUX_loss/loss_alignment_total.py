import torch

# CONFIG:
TAU_S = 0.1
TAU_T = 0.04


def align_loss(f_hat, y, tau_s=TAU_S, tau_t=TAU_T):
    p_y = torch.softmax(y / tau_t, dim=-1)
    log_p_f = torch.log_softmax(f_hat / tau_s, dim=-1)
    return -(p_y * log_p_f).sum(dim=-1).mean()

def loss_alignment_total(c_hat, yc,
                         f_hat, yf,
                         r_hat, yr):
    """
    Compute the total alignment loss for control, turn, and lane predictions.

    Args:
        c_hat (torch.Tensor): The predicted control action from the model (shape: [batch_size, num_classes]).
        yc (torch.Tensor): The ground truth control action labels (shape: [batch_size]).
        f_hat (torch.Tensor): The predicted turn action from the model (shape: [batch_size, num_classes]).
        yf (torch.Tensor): The ground truth turn action labels (shape: [batch_size]).
        r_hat (torch.Tensor): The predicted lane action from the model (shape: [batch_size, num_classes]).
        yr (torch.Tensor): The ground truth lane action labels (shape: [batch_size]).
    Returns:
        torch.Tensor: The computed total alignment loss value.
    """

    current_loss = align_loss(c_hat, yc)
    future_loss = align_loss(f_hat, yf)
    reasoning_loss = align_loss(r_hat, yr)

    total_loss = current_loss + future_loss + reasoning_loss
    return total_loss


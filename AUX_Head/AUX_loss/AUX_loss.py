
# AUX_loss.py
from loss_alignment_total import loss_alignment_total
from loss_action import loss_action

# 权重
lambda1 = 1 # alignment loss权重，我们更重视语义理解
lambda2 = 0.1 # action loss权重，作为辅助监督，帮助模型更好地学习语义与动作之间的关系

def AUX_loss(c_hat, yc,
             f_hat, yf,
             r_hat, yr,
             control_hat, ycontrol,
             turn_hat, yturn,
             lane_hat, ylane):
    """
    Compute the total auxiliary loss for the model, combining alignment and action losses.

    Args:
        c_hat (torch.Tensor): The predicted control action from the model (shape: [batch_size, num_classes]).
        yc (torch.Tensor): The ground truth control action labels (shape: [batch_size]).
        f_hat (torch.Tensor): The predicted turn action from the model (shape: [batch_size, num_classes]).
        yf (torch.Tensor): The ground truth turn action labels (shape: [batch_size]).
        r_hat (torch.Tensor): The predicted lane action from the model (shape: [batch_size, num_classes]).
        yr (torch.Tensor): The ground truth lane action labels (shape: [batch_size]).
        control_hat (torch.Tensor): The predicted control action for action loss (shape: [batch_size, num_classes]).
        ycontrol (torch.Tensor): The ground truth control action labels for action loss (shape: [batch_size]).
        turn_hat (torch.Tensor): The predicted turn action for action loss (shape: [batch_size, num_classes]).
        yturn (torch.Tensor): The ground truth turn action labels for action loss (shape: [batch_size]).
        lane_hat (torch.Tensor): The predicted lane action for action loss (shape: [batch_size, num_classes]).
        ylane (torch.Tensor): The ground truth lane action labels for action loss (shape: [batch_size]).
    Returns:
        torch.Tensor: The computed total auxiliary loss value.
    """
    alignment_loss = loss_alignment_total(c_hat, yc, f_hat, yf, r_hat, yr)
    action_loss = loss_action(control_hat, ycontrol, turn_hat, yturn, lane_hat, ylane)
    total_loss = lambda1 * alignment_loss + lambda2 * action_loss
    return total_loss



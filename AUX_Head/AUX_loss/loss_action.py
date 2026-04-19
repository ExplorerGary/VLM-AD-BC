from torch import nn


def loss_action(pred_action, gt_action):
    """
    Compute the loss for the action prediction.

    Args:
        pred_action (torch.Tensor): The predicted action from the model (shape: [batch_size, num_classes]).
        gt_action (torch.Tensor): The ground truth action labels (shape: [batch_size]).
    
    Returns:
        torch.Tensor: The computed loss value.
    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(pred_action, gt_action)
    return loss
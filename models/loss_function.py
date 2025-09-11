import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    """
    Focal Loss to handle class imbalance.
    """

    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

class CombinedLoss(nn.Module):

    """
    Combined Loss function that includes Focal Loss and Mean Absolute Error.
    """

    def __init__(self, focal_weight=0.5, mae_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.mae_weight = mae_weight
        self.focal_weight = focal_weight

    def forward(self, inputs, targets, regression_targets):
        
        # Convert logits to probabilities for focal loss
        probabilities = torch.sigmoid(inputs)
        focal_loss_value = self.focal_loss(probabilities, targets)
        
        # Calculate MAE for regression
        mae_loss_value = F.l1_loss(inputs, regression_targets)
        
        # Combine the losses
        combined_loss = self.focal_weight * focal_loss_value + self.mae_weight * mae_loss_value
        return combined_loss


if __name__ == "__main__":

    # Example inputs and targets
    inputs = torch.randn(10, 1)  # Model outputs (logits)
    targets = torch.randint(0, 2, (10, 1)).float()  # Ground truth labels for classification
    regression_targets = torch.randn(10, 1)  # Ground truth values for regression

    # Initialize the Combined Loss function
    loss_function = CombinedLoss(focal_weight=0.5, mae_weight=0.5)

    # Calculate the loss
    loss = loss_function(inputs, targets, regression_targets)

    print("Combined Loss:", loss.item())
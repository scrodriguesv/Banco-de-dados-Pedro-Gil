import torch
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import MulticlassPrecision, MulticlassRecall, MulticlassF1Score
import evaluate

class meanIoU(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mIoU = evaluate.load('mean_iou')

    def forward(self, predicted, masks, num_classes):

        self.mIoU.add_batch(
            predictions=predicted.detach().cpu().numpy(), 
            references=masks.detach().cpu().numpy()
        )
        metrics = self.mIoU.compute(
                num_labels=num_classes, 
                ignore_index=255, 
                reduce_labels=False,
            )
        
        return metrics

class MeanAbsoluteError(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        '''
        Calculates the mean absolute error

        Args:
            outputs (_type_): predictions of the model
            labels (_type_): ground truth

        Returns:
            loss: torch tensor of the loss containing the loss and the gradients
        '''

        return F.l1_loss(outputs, labels)

class CategoricalAccuracy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        '''
        Calculates the categorical accuracy

        Args:
            outputs (_type_): predictions of the model
            labels (_type_): ground truth

        Returns:
            accuracy: torch tensor of the accuracy containing the accuracy and the gradients
        '''
        
        outputs = torch.argmax(outputs, dim=1)
        accuracy = (outputs == labels).float().mean()
        return accuracy

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, labels):
        '''
        Calculates the cross entropy loss

        Args:
            outputs (_type_): predictions of the model
            labels (_type_): ground truth

        Returns:
            loss: torch tensor of the loss containing the loss and the gradients
        '''

        return F.cross_entropy(outputs, labels)        

class Precision(torch.nn.Module):
    """
    Computes the weighted precision score across all classes.

    Weighted precision is calculated per class and then averaged using
    the class frequencies in the ground truth labels as weights.
    """

    def __init__(self, num_classes=12):
        """
        Args:
            num_classes (int): Number of target classes. Defaults to 12.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, labels):
        """
        Computes the weighted precision.

        Args:
            outputs (torch.Tensor): Raw logits from the model with shape (N, C, ...).
            labels (torch.Tensor): Ground truth class indices with shape (N, ...).

        Returns:
            torch.Tensor: Scalar tensor with the weighted precision value.
        """
        device = outputs.device  # Get the device from outputs
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            true_positive = torch.zeros(self.num_classes, device=device)  # Move to the same device
            false_positive = torch.zeros(self.num_classes, device=device)  # Move to the same device
            for i in range(self.num_classes):
                true_positive[i] = torch.sum((preds == i) & (labels == i))
                false_positive[i] = torch.sum((preds == i) & (labels != i))
            precision = true_positive / (true_positive + false_positive + 1e-10)
            weighted_precision = torch.sum(precision * torch.bincount(labels.to(torch.int64), minlength=self.num_classes) / len(labels))
        return weighted_precision

class Recall(torch.nn.Module):
    """
    Computes the weighted recall score across all classes.

    Weighted recall is calculated per class and then averaged using
    the class frequencies in the ground truth labels as weights.
    """

    def __init__(self, num_classes=12):
        """
        Args:
            num_classes (int): Number of target classes. Defaults to 12.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, outputs, labels):
        """
        Computes the weighted recall.

        Args:
            outputs (torch.Tensor): Raw logits from the model with shape (N, C, ...).
            labels (torch.Tensor): Ground truth class indices with shape (N, ...).

        Returns:
            torch.Tensor: Scalar tensor with the weighted recall value.
        """
        device = outputs.device  # Get the device from outputs
        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            true_positive = torch.zeros(self.num_classes, device=device)  # Move to the same device
            false_negative = torch.zeros(self.num_classes, device=device)  # Move to the same device
            for i in range(self.num_classes):
                true_positive[i] = torch.sum((preds == i) & (labels == i))
                false_negative[i] = torch.sum((preds != i) & (labels == i))
            recall = true_positive / (true_positive + false_negative + 1e-10)
            weighted_recall = torch.sum(recall * torch.bincount(labels.to(torch.int64), minlength=self.num_classes) / len(labels))
        return weighted_recall

class F1Score(torch.nn.Module):
    """
    Computes the weighted F1 score across all classes.

    F1 score is the harmonic mean of precision and recall. Both metrics are
    computed in their weighted forms using :class:`Precision` and :class:`Recall`.
    """

    def __init__(self, num_classes=12):
        """
        Args:
            num_classes (int): Number of target classes. Defaults to 12.
        """
        super().__init__()
        self.precision_module = Precision(num_classes=num_classes)
        self.recall_module = Recall(num_classes=num_classes)

    def forward(self, outputs, labels):
        """
        Computes the weighted F1 score.

        Args:
            outputs (torch.Tensor): Raw logits from the model with shape (N, C, ...).
            labels (torch.Tensor): Ground truth class indices with shape (N, ...).

        Returns:
            torch.Tensor: Scalar tensor with the weighted F1 score value.
        """
        with torch.no_grad():
            precision = self.precision_module(outputs, labels)
            recall = self.recall_module(outputs, labels)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        return f1_score

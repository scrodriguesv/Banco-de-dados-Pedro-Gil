import torch, os  
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small

from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# SegFormer/Hugging Face

class SegformerFinetuner(nn.Module):
    """Fine-tunable SegFormer model for semantic segmentation.

    Wraps a pretrained SegformerForSemanticSegmentation model loaded from
    local weights, configured for a custom set of labels.

    Args:
        model_weights_path: Path to the directory containing the pretrained model weights.
        id2label: Dictionary mapping class IDs (int) to label names (str).

    Attributes:
        id2label: Dictionary mapping class IDs to label names.
        num_classes: Number of segmentation classes.
        label2id: Reverse mapping from label names to class IDs.
        model: The underlying SegformerForSemanticSegmentation instance.
    """
    def __init__(self, model_weights_path, id2label):
        super(SegformerFinetuner, self).__init__()

        self.id2label = id2label
        self.num_classes = len(id2label.keys())
        self.label2id = {v:k for k,v in self.id2label.items()}

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_weights_path,
            local_files_only=True, 
            return_dict=False, 
            num_labels=self.num_classes,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)


def image_processor_starter(processor_path: os.PathLike) -> SegformerImageProcessor:
    """Load and configure a SegformerImageProcessor from a local path.

    Sets the processing size based on the model variant:
    - 'b0' models use size 128
    - All other models use size 1024

    Args:
        processor_path: Path to the directory containing the processor files.

    Returns:
        A configured SegformerImageProcessor instance.
    """
    processor = SegformerImageProcessor.from_pretrained(processor_path, local_files_only=True)
    if 'b0' in str(processor_path):
        processor.size = 128
    else:
        processor.size = 1024

    return processor

# # ------------------------------------------------------------------------------------------------------

# class MobileNetV3(nn.Module):
#     def __init__(self, weights_path:os.PathLike=None, freeze:bool=True, num_classes:int=12):
#         super(MobileNetV3, self).__init__()
#         self.model = mobilenet_v3_small()
#         if weights_path:
#             weights = torch.load(weights_path)
#             self.model.load_state_dict(weights)
#         self.model.classifier[-1] = nn.Linear(1024, num_classes)
#         if freeze:
#             if not weights_path:
#                 raise ValueError('You must provide the path to the weights file if you want to freeze the model.')
#             for idx, child in enumerate(self.model.children()):
#                 if idx < 2:
#                     for param in child.parameters():
#                         param.requires_grad = False
        
#     def forward(self, x):
#         return self.model(x)

# # ------------------------------------------------------------------------------------------------------

# class CustomCNN(nn.Module):
#     def __init__(self):
#         super(CustomCNN, self).__init__()
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(num_features=64)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
#         self.bn2 = nn.BatchNorm2d(num_features=64)
        
#         self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
#         self.bn3 = nn.BatchNorm2d(num_features=64)
        
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)
#         self.bn4 = nn.BatchNorm2d(num_features=128)
        
#         self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
#         self.bn5 = nn.BatchNorm2d(num_features=128)
        
#         # Dense (fully connected) layers
#         self.fc1 = nn.Linear(in_features=128 * 6 * 6, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=128)
#         self.fc3 = nn.Linear(in_features=128, out_features=64)
#         self.fc4 = nn.Linear(in_features=64, out_features=12)
        
#     def forward(self, x):
#         # Apply convolutions, batch normalization, max pooling and ReLU activation
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = self.pool(F.relu(self.bn5(self.conv5(x))))
        
#         # Flatten the output for the dense layers
#         x = x.view(-1, 128 * 6 * 6)
        
#         # Apply dense layers with ReLU activation, except for the output layer
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)  # No activation function for the output layer
        
#         return x


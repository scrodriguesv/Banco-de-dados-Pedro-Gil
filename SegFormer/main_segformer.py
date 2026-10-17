import torch, os

# Boilerplate
from source.dataset import SemanticSegmentationDataset
from source.losses import meanIoU
from source.project_segformer import Project
from source.callbacks import Logger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Tensorboard
from source.neural_network import SegformerFinetuner, image_processor_starter

# DSETS: [] dset1 [] dset2 [] dset3 [] dset4 [] dset5 
# WEIGHTS: [] b0 [] b5
# DONE: [x] dset1-b0  [x] dset2-b0  [x] dset3-b0  [x] dset4-b0  [x] dset5-b0 
# DONE: [x] dset1-b5 [x] dset2-b5 [x] dset3-b5 [x] dset4-b5 [x] dset5-b5

# Configs
basic_dir = '/tatu/dxvw/projetos/edge_corrosion/final_model/final_model'
# CHANGE HERE
fold_dir = 'dset4'

root_dir = os.path.join(basic_dir, fold_dir)

b0_weights_path = '/tatu/dxvw/projetos/edge_corrosion/final_model/final_model/pretrained_models/segformer-b0-finetuned-cityscapes-1024-1024'
b5_weights_path = '/tatu/dxvw/projetos/edge_corrosion/final_model/final_model/pretrained_models/segformer-b5-finetuned-cityscapes-1024-1024'
# CHANGE HERE
chosen_weights_path = b5_weights_path

basic_save_path = '/tatu/dxvw/projetos/edge_corrosion/final_model/final_model/trained_models'
base_save_path = os.path.join(basic_save_path, fold_dir)

# Datasets
processor = image_processor_starter(chosen_weights_path)
train_dataset = SemanticSegmentationDataset(root_dir=root_dir, specific_path='train', feature_extractor=processor)
val_dataset = SemanticSegmentationDataset(root_dir=root_dir, specific_path='val', feature_extractor=processor)

# Checkpoints
model_save_path = os.path.join(base_save_path, chosen_weights_path.split('/')[-1])
# if not os.path.exists(model_save_path):
#     os.makedirs(model_save_path)
print(f'\n\nTrain dataset path: {os.path.join(root_dir, "train")}\n\n')
print(f'\n\nVal dataset path: {os.path.join(root_dir, "val")}\n\n')
print(f'\n\nSave path: {model_save_path}\n\n')

model = SegformerFinetuner(model_weights_path=chosen_weights_path, id2label=train_dataset.id2label)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-05, eps=1e-08)

logger = Logger()
scheduler = ReduceLROnPlateau(patience=5, factor=0.5, verbose=True, min_delta=1e-4)
early_stopping = EarlyStopping(patience=30, verbose=True, min_delta=1e-4)
checkpoint = ModelCheckpoint(save_path=model_save_path)
tensorboard = Tensorboard(user='dxvw', name='edge_corrosion')

# callbacks = [logger, scheduler, early_stopping]
callbacks = [logger, checkpoint, tensorboard, early_stopping, scheduler]
# metrics = [CrossEntropyLoss(), Precision(), Recall(), F1Score()]
metrics = [meanIoU()]

project = Project(num_classes=model.num_classes)
project.compile(model)

history = project.fit(
    train_dataset,
    val_dataset,
    epochs=1000,
    batch_size=4,
    optimizer=optimizer,
    callbacks=callbacks,
    metrics = metrics,
    learning_rate = 0.001,
)

import torch, os, datetime

# Boilerplate
from source.dataset import CustomDataset
from source.losses import CrossEntropyLoss, CategoricalAccuracy
from source.project import Project
from source.callbacks import Logger, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, Tensorboard
from source.neural_network import MobileNetV3

train_dataset = CustomDataset(df_path='/pgeoprj/dxvw/edge_corrosion/dataset/kfold/first/train_4.csv', src_path='/pgeoprj/dxvw/edge_corrosion/dataset/')
val_dataset = CustomDataset(df_path='/pgeoprj/dxvw/edge_corrosion/dataset/kfold/first/valid_4.csv', src_path='/pgeoprj/dxvw/edge_corrosion/dataset/')

base_save_path = '/pgeoprj/dxvw/edge_corrosion/models/'
now_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
model_save_path = os.path.join(base_save_path, f'fold4/') 
print(f'\n\nSave path: {model_save_path}\n\n')
# if not os.path.exists(model_save_path):
#     os.makedirs(model_save_path)

loss = CrossEntropyLoss()
mobile_weights = '/pgeoprj/dxvw/edge_corrosion/models/mobilenetv3_small/mobilenet_v3_small-047dcff4.pth'
model = MobileNetV3(mobile_weights, freeze=True, num_classes=12)
optimizer = torch.optim.RMSprop(model.parameters())
# optimizer = torch.optim.Adam(model.parameters())
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

logger = Logger()
scheduler = ReduceLROnPlateau(patience=5, factor=0.5, verbose=True, min_delta=1e-4)
early_stopping = EarlyStopping(patience=12, verbose=True, min_delta=1e-4)
checkpoint = ModelCheckpoint(save_path=model_save_path)
tensorboard = Tensorboard(user='dxvw', name='edge_corrosion')

# callbacks = [logger, scheduler, early_stopping]
callbacks = [logger, checkpoint, tensorboard, early_stopping, scheduler]
# metrics = [CrossEntropyLoss(), Precision(), Recall(), F1Score()]
metrics = [CategoricalAccuracy()]

project = Project()
project.compile(model)

history = project.fit(
    train_dataset,
    val_dataset,
    epochs=100,
    batch_size=100,
    optimizer=optimizer,
    loss=loss,
    callbacks=callbacks,
    metrics = metrics,
    learning_rate = 0.001,
)

# from article - 30 epochs  | batch size 100 | learning rate 0.001  | momentum 0.9;
# default      - 100 epochs | batch size 32  | learning rate 0.0025 | momentum 0.9;
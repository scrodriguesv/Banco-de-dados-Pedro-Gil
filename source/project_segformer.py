import torch
from source.callbacks import CallbackHandler
import numpy as np
import torch.nn.functional as F
import source.distributed_utils as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from transformers import SegformerImageProcessor

class Project():
    '''
    A keras-like class to control your project.
    The most high level class of the project.
    Everything else should be changed in the dependencies.
    This class creates the training loop and handles the callbacks.
    Use callbacks to change the training loop, like changing the learning rate or saving the model.
    '''

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.rank, self.world_size, self.gpu = dist.init_distributed_mode()
        dist.fix_random_seeds()
        cudnn.benchmark = True # algorithm optimization

    def compile(self, model):
        '''
        Compiles the model by converting BatchNorm layers to SyncBatchNorm and wrapping the model in DDP.
        For more information on SyncBatchNorm, see https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html

        Args:
            model (torch.nn.Module): your neural network model
        '''
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.model = DDP(self.model.to(self.gpu), device_ids=[self.gpu])

    def predict(self, inputs):
        '''
        Method to predict the output of the model for a single input batch.

        Args:
            inputs (PIL.Image): input image to be segmented

        Returns:
            (_type_): output of the model
        '''
        
        processor = SegformerImageProcessor(do_resize=False)
        self.model.eval()
        with torch.no_grad():
            pixel_values = processor(inputs, return_tensors="pt").pixel_values.float().to(self.gpu)
            outputs = self.model(pixel_values)
            predicted_segmentation_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[inputs.size[::-1]])[0]
            predicted_segmentation_map = predicted_segmentation_map.cpu().numpy()
        return outputs
    

    def evaluate(self, val_loader):
        '''
        Method for model evaluation of validation loss and metrics

        Args:
            val_loader (Dataloader): data loader for the test set

        Returns:
            float: validation loss
            list of floats: list of validation metrics
        '''
        # FIND A WAY OF GETTING ALL METRICS
        val_loss = torch.tensor(0.0).to(self.gpu)
        metric_loss = torch.zeros(3 + 2*(self.num_classes)).to(self.gpu)
        self.model.eval()

        with torch.no_grad():
            for img_lbl_pair in val_loader:

                with autocast():
                    # forward pass
                    inputs = img_lbl_pair['pixel_values'].to(self.gpu).float()
                    targets = img_lbl_pair['labels'].to(self.gpu)
                    outputs = self.model(inputs, targets)
                    
                    # calculate loss
                    loss, logits = outputs[0], outputs[1]
                    val_loss += loss

                upsampled_logits = torch.nn.functional.interpolate(
                        logits, 
                        size=targets.shape[-2:], 
                        mode="bilinear", 
                        align_corners=False
                    )

                predicted = upsampled_logits.argmax(dim=1)

                calculated_metrics = self.metrics[0](predicted, targets, self.num_classes)

                metrics_list = []
                for _, value in calculated_metrics.items():
                    if type(value)==np.ndarray:
                        for element in value:
                            metrics_list.append(element)
                    else:
                        metrics_list.append(value)
                
                metric_loss += torch.tensor(metrics_list).to(self.gpu)
                
            # average loss and metrics
            metric_loss /= len(val_loader)
            val_loss /= len(val_loader)

        return val_loss, metric_loss

    def fit(self,
            train_dataset,
            val_dataset=None,
            optimizer=None,
            epochs=5,
            batch_size=1,
            shuffle=True,
            learning_rate=0.001,
            callbacks=[],
            metrics=[],
            ):
        '''
        Method to train the model.
        Use this to change the training loop if needed.
        Notice that any change that is not structural should be done using callbacks.
        For example, if you want to change the learning rate during training, you should use a callback.
        Otherwise, if you are training for example a GAN, you should change the training loop.

        Args:
            train_dataset (Dataset): dataset for the training set.
            val_dataset (Dataset optional): dataset for the evaluation set. Defaults to None. If None, the training set is used.
            optimizer (optimizer, optional): optimizer used to train the model. Defaults to None.
            epochs (int, optional): number of epochs to train the model for. Defaults to 10.
            batch_size (int, optional): batch size for training. Defaults to 1.
            shuffle (bool, optional): whether to shuffle or not the data before training and testing. Defaults to True.
            learning_rate (float, optional): learning rate. Defaults to 0.001.
            callbacks (list of Callbacks, optional): callbacks to run during trainig. Defaults to [].
            metrics (list of metrics, optional): metrics to use for evaluation. Defaults to [].
        
        Returns:
            dict: dictionary containing history of the training
        '''

        train_sampler = DistributedSampler(train_dataset, shuffle=shuffle)
        self.train_loader = DataLoader(train_dataset,sampler=train_sampler, batch_size=batch_size, pin_memory=True)
        if (val_dataset != None):
            val_sampler = DistributedSampler(val_dataset, shuffle=shuffle)
            self.val_loader = DataLoader(val_dataset,sampler=val_sampler, batch_size=batch_size, pin_memory=True)
        else:
            self.val_loader = DataLoader(train_dataset,sampler=train_sampler, batch_size=batch_size, pin_memory=True)
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.callbacks = callbacks
        self.metrics = metrics

        self.optimizer.param_groups[0]['lr'] = self.learning_rate

        self.train_loss = []
        self.val_loss = []
        self.metrics_loss = []
        self.epoch_loss = torch.tensor(0.0).to(self.gpu)
        self.stop_training = False

        callback_handler = CallbackHandler(callbacks, learner=self)
        callback_handler.on_train_begin()

        scaler = GradScaler()

        for self.epoch in range(1, epochs+1):
            callback_handler.on_epoch_begin()
            self.model.train()
            
            for self.batch, img_lbl_pair in enumerate(list(self.train_loader)):
                callback_handler.on_batch_begin()

                with autocast():
                    # forward pass
                    inputs = img_lbl_pair['pixel_values'].float().to(self.gpu)
                    targets = img_lbl_pair['labels'].to(self.gpu)
                    outputs = self.model(inputs, targets)

                    # calculate loss
                    self.batch_loss = outputs[0]

                # backward pass
                self.optimizer.zero_grad()
                scaler.scale(self.batch_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                torch.distributed.all_reduce(self.batch_loss, op=torch.distributed.ReduceOp.AVG)
                self.epoch_loss += self.batch_loss
                
                callback_handler.on_batch_end()

            val, metric = self.evaluate(self.val_loader)
            torch.distributed.all_reduce(val, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(metric, op=torch.distributed.ReduceOp.AVG)
            
            self.train_loss.append((self.epoch_loss).item()/len(self.train_loader))
            self.val_loss.append(val.item())
            self.metrics_loss.append(metric.cpu().numpy().tolist())

            self.epoch_loss = torch.tensor(0.0).to(self.gpu)
            
            callback_handler.on_epoch_end()

            if (self.stop_training): break

        callback_handler.on_train_end()
        self.history = {'train_loss': self.train_loss, 
                        'val_loss': self.val_loss,
                        'metrics': self.metrics_loss}

        return self.history

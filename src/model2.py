import torch
from pytorch_lightning import LightningModule
from torchvision.models import resnet50
# from torchvision.models import DeepLabV3_ResNet50_Weights
# from torchvision.models import DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_resnet101
from transformers import SegformerForSemanticSegmentation
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import numpy as np
import torch.nn as nn
import cv2
from PIL import Image
import wandb
from src.metric import  mean_iou, mean_dice
from src.visualize import plot_eval
from kornia.losses import FocalLoss
from kornia.losses import LovaszSoftmaxLoss

def map_index(x):
    mapindex = {0:0,1:5,2:3,3:9,4:10,5:11,6:12,7:13,8:8,9:2,10:14,11:15,12:16,13:17,14:1,15:7,16:18,17:19,18:20,19:21,20:4,21:22,22:23,23:24,24:25,25:26,26:6,27:27}
    return mapindex[x]

class _segmentation_model(LightningModule):
    def __init__(self, model, loss_fn, learning_rate, scheduler_patience=4, scheduler_factor=0.5):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.scheduler_patience = scheduler_patience
        self.scheduler_factor = scheduler_factor
    
    def training_step(self, batch, batch_idx):
        x, y= batch
        logits = self(x).squeeze()
        if list(logits.shape)[-1] == 64:
            logits = nn.Upsample([256,256], mode='bilinear', align_corners=False)(logits)
        loss = self.loss_fn(logits, y.long())
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x).squeeze()
        if list(logits.shape)[-1] == 64:
            logits = nn.Upsample([256,256], mode='bilinear', align_corners=False)(logits)
        y = y.squeeze()
        loss = self.loss_fn(logits, y.long())
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.scheduler_factor, patience=self.scheduler_patience)
        return {"optimizer": optimizer, "monitor": "val_loss", "lr_scheduler": scheduler,}


class _single_task_segmentation(_segmentation_model):
    def __init__(self, model, loss_fn, learning_rate,num_classes=9):
        super().__init__(model, loss_fn, learning_rate)
        self.model = model
        self.loss_fn = loss_fn
        self.num_classes = num_classes
        self.test_step_outputs = []


    def test_step(self, batch, batch_idx):
        x, y = batch
        y = y.squeeze().cpu()
        pred = self(x)
        pred = nn.Upsample([256,256], mode='bilinear', align_corners=False)(pred)
        pred = pred.argmax(1).cpu()
        pred_iou = pred.detach().clone()
        pred_iou = pred_iou.apply_(map_index)
        pred_iou[pred_iou==11]=3
        pred_iou[pred_iou==15]=2
        pred_iou[pred_iou==7]=4
        pred_iou[pred_iou==8]=7
        pred_iou[pred_iou > 7] = 0

        y[y==8] = 7

        self.test_step_outputs.append((pred_iou.squeeze(), y.squeeze()))
        return (pred, y)

    def on_test_epoch_end(self):
        preds = []
        ys = []
        for pred_batch, y_batch in self.test_step_outputs:
            preds.extend(pred_batch)
            ys.extend(y_batch)
        _, _, iou = mean_iou(preds, ys, ignore_index=-1, num_classes=self.num_classes)
        _,_,dice = mean_dice(preds, ys, ignore_index=-1, num_classes=self.num_classes)
        classes = ["Hood", "Front door", "Rear door", "Frame", "Rear quarter panel", "Trunk lid", "Bumper"]
        for u,i in enumerate(classes):
            self.log("mIoU "+i, iou[u+1], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log("dice "+i, dice[u+1], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("mIoU", np.nanmean(iou[1:]), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("Dice", np.nanmean(dice[1:]), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        #self.log(f"{name} miou_w_bg", np.mean(iou), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, im = batch
        preds = self(x).sigmoid()
        binary_preds = (preds > 0.5).int()
    

class deeplab(_single_task_segmentation):
    def __init__(self, learning_rate, num_classes, backbone="resnet50"):
        loss_fn = CrossEntropyLoss()
        #loss_fn = FocalLoss(alpha = 0.5, reduction = "mean")
        if backbone == "resnet50": 
            model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        else:
            model = deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
        model.classifier[-1] = torch.nn.Conv2d(
                256, num_classes, kernel_size=(1, 1), stride=(1, 1)
            )
        super().__init__(model=model, loss_fn=loss_fn, learning_rate=learning_rate)
        self.model = model

    def forward(self, x):
        x = self.model(x)["out"]
        return x

class segformer2(_single_task_segmentation):
    def __init__(self, learning_rate, num_classes):
        #loss_fn = CrossEntropyLoss()
        loss_fn = FocalLoss(alpha = 0.5, reduction = "mean")
        #loss_fn = LovaszSoftmaxLoss()
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        model.decode_head.classifier = torch.nn.Conv2d(768,num_classes,kernel_size=(1,1), stride=(1,1))
        super().__init__(model=model, loss_fn=loss_fn, learning_rate=learning_rate,num_classes=num_classes)
        self.model = model

    def forward(self, x):
        x = self.model(x)["logits"]
        return x
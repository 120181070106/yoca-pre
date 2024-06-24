import os

import torch
from tqdm import tqdm
import torch.nn.functional as F
from utils.utils import get_lr
def reg_l1_loss(pred, target, mask): #在首函前添加这两函数
    pred = pred.permute(0,2,3,1)
    expand_mask = torch.unsqueeze(mask,-1).repeat(1,1,1,2)
    loss = F.l1_loss(pred * expand_mask, target * expand_mask, reduction='sum')
    loss = loss / (mask.sum() + 1e-4)
    return loss
def focal_loss(pred, target):
    pred = pred.permute(0, 2, 3, 1)
    #-------------------------------------------------------------------------#
    #   找到每张图片的正样本和负样本
    #   一个真实框对应一个正样本
    #   除去正样本的特征点，其余为负样本
    #-------------------------------------------------------------------------#
    pos_inds = target.eq(1).float()
    neg_inds = target.lt(1).float()
    #-------------------------------------------------------------------------#
    #   正样本特征点附近的负样本的权值更小一些
    #-------------------------------------------------------------------------#
    neg_weights = torch.pow(1 - target, 4)
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)
    #-------------------------------------------------------------------------#
    #   计算focal loss。难分类样本权重大，易分类样本权重小。
    #-------------------------------------------------------------------------#
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    #-------------------------------------------------------------------------#
    #   进行损失的归一化
    #-------------------------------------------------------------------------#
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0
    cls_loss = 0
    loc_loss = 0#补入两预测损失

    if local_rank == 0:
        # print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, bboxes, hms, whs, masks = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
                hms = hms.cuda(local_rank)
                whs = whs.cuda(local_rank)
                masks = masks.cuda(local_rank)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        if not fp16:
            #----------------------#
            #   前向传播
            #----------------------#
            # dbox, cls, origin_cls, anchors, strides 
            outputs = model_train(images)
            loss_value = yolo_loss(outputs, bboxes)
            #----------------------#
            #   反向传播
            #----------------------#
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                #----------------------#
                #   前向传播
                #----------------------#
                outputs         = model_train(images)
                loss_main = yolo_loss(outputs[:7], bboxes)
                loss_cls = focal_loss(outputs[-2], hms)
                loss_loc = reg_l1_loss(outputs[-1], whs, masks)*0.1
                loss_value = loss_main+loss_cls+loss_loc#替原y_loss(outputs,bboxes)

            #----------------------#
            #   反向传播
            #----------------------#
            scaler.scale(loss_value).backward()
            scaler.unscale_(optimizer)  # unscale gradients
            torch.nn.utils.clip_grad_norm_(model_train.parameters(), max_norm=10.0)  # clip gradients
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        cls_loss += loss_cls.item()
        loc_loss += loss_loc.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss':loss/(iteration+1),'中':cls_loss/(iteration+1),'尺':loc_loss/(iteration+1),'lr':get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()
        
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, bboxes = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
            #----------------------#
            #   清零梯度
            #----------------------#
            optimizer.zero_grad()
            #----------------------#
            #   前向传播
            #----------------------#
            outputs     = model_train_eval(images)[:7]
            loss_value  = yolo_loss(outputs, bboxes)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        # print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        # print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            # print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "b.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "l.pth"))
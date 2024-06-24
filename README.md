```
#---------------------------相对的改进，如需训练先执行annotation.py划分出本地的图片集合，纯推理预测则不用------------------------#
#---------------------------(predict.ipynb)------------------------#
    if mode == "predict":
        image = Image.open('4.jpg')#先自动对目录下的4.jpg文件实施基线预测
        #此外还提供的基准图片有：45是基线目标，67是大目标，89是小目标，cd是难目标
        r_image = yolo.detect_image(image, crop = crop, count=count)
        r_image.show()
        while True:
            img = input('Input image filename:')
            try:#这样自动叠加后缀就只需要输入文件名
                image = Image.open(img+'.jpg')
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()
#---------------------------(yolo.py)------------------------#
        "model_path"        : 'model_data/b基础633.pth',#原yolov8_s换为自训的基线权
        "classes_path"      : 'model_data/voc_classes.txt',#只含0到6七类，分别分行
        "phi"               : 'n',#版本从s换为更易训、内存更小的n 
        "cuda"              : False,#cuda换为否方便推理时切无卡模式用cpu更省钱
#---------------------------(utils_fit.py)------------------------#
    if local_rank == 0:#去掉开训和完训，以及验证全程的显示
        # print('Start Train')
    if local_rank == 0:
        pbar.close()
        # print('Finish Train')
        # print('Start Validation')
        # pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    if local_rank == 0:
        pbar.close()
        # print('Finish Validation')
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):#关掉最优权的保存提示，将定期权重名改为p030三个数的形式，忽略具体损失，最后精简best_epoch_weights为b，last_epoch_weights为l
            torch.save(save_state_dict, os.path.join(save_dir, "p%03d.pth" % (epoch + 1)))
            # torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))
            # print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "b.pth"))
        #     torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
        # torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
        torch.save(save_state_dict, os.path.join(save_dir, "l.pth"))
#---------------------------(callbacks.py)------------------------#
            # print("Calculate Map.")
            # print("Get map done.") #关掉算map始末的提示
#---------------------------(train.ipynb)------------------------#
if __name__ == "__main__": #精简参数行，去除多余注释
    Cuda            = True #服务器训练只能用gpu，无卡模式cpu训不了
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True #设true更快些
    classes_path    = 'model_data/voc_classes.txt'
    model_path      = 'b基础633.pth' #原为'model_data/yolov8_s.pth'改成咱们自训的
    input_shape     = [640, 640]
    phi             = 'n' # 原's'改更小更高效
    pretrained      = False #有权重就不用预训练
    mosaic              = True
    mosaic_prob         = 0.5
    mixup               = True
    mixup_prob          = 0.5
    special_aug_ratio   = 0.7
    label_smoothing     = 0
    Init_Epoch          = 0
    Freeze_Epoch        = 50
    Freeze_batch_size   = 2 #原32改小
    UnFreeze_Epoch      = 300
    Unfreeze_batch_size = 4 #原16改小
    Freeze_Train        = False #预冻结前50的骨网权重，在前置网需要同时训练故设False
    Init_lr             = 1e-2
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4
    lr_decay_type       = "cos"
    save_period         = 30 #每隔30轮保存下权重，整个只需10个文件，减少原10的冗余
    save_dir            = 'logs'
    eval_flag           = True
    eval_period         = 10
    num_workers         = 4```
预处理改进
```
#跑训练前先执行annotation.py生成本地数据集，改动涉及：
# net:yolo+backbone+yolo_training
# utils:utils_bbox+dataloader
# yolo
#-----------------------------(backbone.py)------------------------#
# import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.enabled = False
import os,torchvision,random##最初506
from torchvision.utils import save_image

class Backbone(nn.Module): 
    # self.stem = Conv(3, base_channels, 3, 2)
    self.pro = nn.Sequential(Conv(3,16,5,1),Conv(16,32,7,1),Conv(32,64,9,1),Conv(64,32,7,1),Conv(32,16,5,1))#5+7+9+7+5+3(后两)=46占约160四一,sig初随写防后复写
    self.cen=nn.Sequential(nn.Conv2d(16,1,3,1,padding=1))
    self.wh=nn.Sequential(nn.Conv2d(16,2,3,1,padding=1))#此处还可去Sequential简化

def hot(self, x):#同forward一起替换原来的forward，注意统一制表一下
    路口图=self.pro(F.interpolate(x,(160,160)))#下采减算扩野,预转为小图尺度
    return self.cen(路口图).sigmoid(),self.wh(路口图).sigmoid()*160
def forward(self, x):#注意sqz(0)在0维非1时失效!,先定义后面常用表达
    中心图,尺度图=self.hot(x);批数=x.shape[0];原图=x.clone(); 在测=批数==1
    元=torch.ones(批数,2);目高宽,偏移量,放缩量=元.clone(),元.clone()*0,元.clone()
    for i in range(批数):#无热点则拆作2*2入网细检再拼 range(0,批数,2)
        if pr(中心图[i])==1 and 在测:#nms核大小只在中心图无热点时才定义为2,即说明过小
            分区图=torch.stack((x[i,:,0:320,0:320],x[i,:,0:320,320:640],x[i,:,320:640,0:320],x[i,:,320:640,320:640]),dim=0); print("过小!")
            中心图[i],尺度图[i]=cat(self.hot(分区图)[0]),cat(self.hot(分区图)[1])
        点尺图=nmask(中心图[i],pr(中心图[i]))*尺度图[i]; 基=-1#除以四转小图尺
        目高宽,均尺度=avg(点尺图);板=x[0].unsqueeze(0).repeat(2,1,1,1)*0+0.5
        if 均尺度>32 and 在测:#均尺>32接近单点感受野多占比大目标故强缩于心,两黑图打底
            a=slice(160,480); b=slice(240,400); print("过大!")#x防转cu
            板[0,:,a,a]=rez(x[i],320); 板[1,:,b,b]=rez(x[i],160); 上均尺=均尺度
            准心图,准尺图=self.hot(板);准点图=torch.zeros(2,2,160,160)
            for k in [0,1]:#考虑到不可能无限缩小,故在每次取更小图时加约束
                准点图[k]=nmask(准心图[k],pr(准心图[k]))*准尺图[k]
                准高宽,准均尺=avg(准点图[k])#预填原均尺给-1,只更前两,保清晰度
                if (准均尺*2>上均尺+3)and(torch.max(准心图[k])).item()>0.1:
                    目高宽=准高宽;基=k;点尺图=准点图[k];x[i]=板[k];上均尺=准均尺
        十字图=spread(点尺图); 下,上,左,右=pr(十字图)#少了首一维的十字改求核为求边
        x[i],偏移量[i,0],放缩量[i,0]=tpr(x[i],下*4,上*4,目高宽[0],3)#纵调后横调
        x[i],偏移量[i,1],放缩量[i,1]=tpr(x[i],左*4,右*4,目高宽[1],4)
        放缩量[i]*=[0.5,0.25,1][基];偏移量[i]+=[320,960,1][基]*放缩量[i]
        if 在测:imsave(原图,十字图,x,中心图,2)#结合大目标的缩中
    if random.random()<0.05:imsave(原图,十字图,x,中心图) #训练时改为 0.05
    # x=原图;偏移量=元*0;放缩量=元#注意因尺度为640,tpr画板即照此,入尺与界同比,画板上的
    feat1 = self.dark3(self.dark2(self.stem(x)))#偏移量基入尺与界故均音放大四倍到640
    feat2 = self.dark4(feat1); feat3 = self.dark5(feat2)
    return feat1,feat2,feat3,偏移量,放缩量,rez(中心图),rez(尺度图*4)#这个地方转入尺

def imsave(原图,十字图,调后图,中心图,m=1):#这一系列函数直接复制到py文件末尾
    十字掩=rez((十字图.unsqueeze(0).repeat(3,1,1)),640)
    绿布=torch.zeros(3,640,640).to(原图.device); 绿布[1,:,:]=0.8
    加十图=原图[-1]*(1-十字掩)+十字掩*绿布#if a.is_cuda: b = b.cuda()的精简版
    c=torch.cat((rez(加十图,320),F.interpolate(中心图[-1].repeat(3,1,1).unsqueeze(0),(320,160)).squeeze(0),rez(调后图[-1],320)),dim=2)
    image = torchvision.transforms.ToPILImage()((c*255).byte())
    image.save('logs/ctt.png') if m==2 else image.save('logs/com.png')
def nmask(热图,核大小,置信阈=0.3):#热图nms,最池再等上原图稀疏以相隔至少核大小的山峰
    真核 = max(int(核大小.item()),3)#将最小的1转为有意义的3, 设置置信阈字典↓
    置信阈={3:0.1,5:0.2,7:0.3,9:0.2}.get(真核)#大降得少因所防的大目标毕竟特例
    if torch.max(热图)<0.3: 置信阈=0.1#防止热图不够热而被中核升阈而得零
    hmax=F.max_pool2d(热图,(真核,真核),1,(真核-1)//2)
    return ((热图*(hmax==热图).float())>置信阈).float()#大于上置信阈即掩
def pr(热图,模式=1):#(1,640,640)热力图得最适nms核,(640,640)得四个边界,形均(b,)
    if 热图.dim()==3: 模式=0; 热图=热图.squeeze(0)#先取概率可观的点转1否则0
    信掩=(热图>0.1).float(); 下,上,左,右,核=0,0,0,0,[]#先取概率可观的点转1否则0
    纵向和=torch.sum(信掩,0);横向和=torch.sum(信掩,1)#下若不括0时返核,上..
    if 纵向和.sum()==0:核.append(1)#无热点而略过,保留默认的零,核则最细的1
    else:#as_tuple=False表是否将坐标按位拆组元组返; torch.ones_like(受仿矢)
        纵效号=torch.nonzero(纵向和); 横效号=torch.nonzero(横向和)
        下=纵效号[0];上=纵效号[-1]; 左=横效号[0];右=横效号[-1]; w=右-左;h=上-下
        核.append(int(torch.max(torch.stack((w,h),dim=0))/160*4)*2+3)#单素代入坐标后为空不可cat同维拼而stack可,max只返最值,加维度才返索引,argmax只返拉平后索引用,也可定维返同max.如max([[1,5],[2,9]])→9;min([[4,1,5],[2,8,9]],dim=0)→values=[2,1,5],indices=[1,0,0]此处需两量承接;argmax([[4,1,5],[2,8,9]])→5
    return torch.Tensor(核) if 模式==0 else (下,上,左,右)#核按热区占比分四级:3/5/7/9
# 下=torch.nonzero(torch.sum(信掩,0))[0]
def spread(点尺图): #将高宽图分别以热点为中心单向扩散出热点处值的距离
    x,y=点尺图[0],点尺图[1]
    if len(torch.nonzero(x))<100:#100以下才扩散,防止非极抑失效造成过多热点致无限循环
        for 效号 in torch.nonzero(x):#0表示横向1表示纵向,若循环如for i in (首,尾)慢
            横扩度=int(x[效号[0],效号[1]]); 纵扩度=int(y[效号[0],效号[1]])
            x[效号[0],max(效号[1]-横扩度,0):min(效号[1]+横扩度+1,160)] = 1
            y[max(效号[0]-纵扩度,0):min(效号[0]+纵扩度+1,160),效号[1]] = 1
    return torch.clip(x+y,max=1)#若不限于图中,则若出大界无值,出小界会在大界那边补出来
def tpr(x,热区下界,热区上界,目标尺度,模式=3):#(3,640,640)图热区中心化&目标尺度无偏化
    入图尺度,无偏尺度=640,30#热图中无置信点时尺度默认无偏,全图当作热区 
    if 目标尺度*(热区上界-热区下界)==0: 目标尺度=无偏尺度;热区下界=0;热区上界=入图尺度
    放缩量=min(无偏尺度/目标尺度,入图尺度/(热区上界-热区下界-1))#不能放超入图,-1防bug
    偏移量=入图尺度/2-放缩量*(热区下界+热区上界)/2#先原地放缩再偏移,故为放区中点到入中
    热左在图=偏移量>0; 热右在图=放缩量*入图尺度+偏移量<入图尺度#按默认放偏后是否仍图
    仅左在图=热左在图*(not 热右在图); 仅右在图=热右在图*(not 热左在图)
    if (放缩量<1)*仅左在图 or(放缩量>1)*仅右在图:偏移量=(1-放缩量)*入图尺度#△见后
    if (放缩量<1)*仅右在图 or(放缩量>1)*仅左在图:偏移量=0#前应右移左超量:偏-(偏)=0
    if 模式==4:x=x.transpose(1,2)#默认对末维即横向操作,先尽管将入图横向放缩至预定量
    x=F.interpolate(x.unsqueeze(0),(640,int(640*放缩量))).squeeze(0)# ↑
    画板=torch.ones(3,640,640)*0.5#切记不能127当灰因原图已归,会致预测无激只归忽略前景
    if 偏移量<0:画板[:,:,0:int(min(640*放缩量,640-偏移量))-int(-偏移量)]=x[:,:,int(-偏移量):int(min(640*放缩量,640-偏移量))]#x先放缩故右始量为负偏量,偏量为负见表达式为放,若未露出右界则终于640加上绝对偏量,露则限于(640*放缩量),详见示意图
    else:画板[:,:,int(偏移量):int(min(640*放缩量,640-偏移量))+int(偏移量)]=x[:,:,0:int(min(640*放缩量,640-偏移量))]#右偏则原图左界仍图即右从零始,至放图的640减偏量
    if 模式==4:画板=画板.transpose(1,2)#纵向操作的化转置回来
    return 画板,偏移量,放缩量#△应左移右超量:偏-(放*尺+偏-尺)=(1-放)尺,放时右移右空亦
def rez(x,s=640): return F.interpolate(x,(640,640)) if x.dim()==4 else F.interpolate(x.unsqueeze(0),(s,s)).squeeze(0)
def avg(x):#防效数为零时取极值而补乘
    效数=len(torch.nonzero(x[0]))/int(len(torch.nonzero(x[0]))!=0)
    return torch.stack((x[0].sum(),x[1].sum()))/效数,((x[0]+x[1]).sum())/(2*效数)
def cat(x): return rez(torch.cat((torch.cat((x[0],x[2]),dim=1),torch.cat((x[1],x[3]),dim=1)),dim=2),160)#上两和下两沿1高再沿2宽拼,以重生界核及更清尺度

#-----------------------------(yolo.py)------------------------#
class YoloBody(nn.Module):
        # self.stride     = torch.tensor([256 / x.shape[-2] for x in self.backbone.forward(torch.zeros(1, 3, 256, 256))])  # forward
        self.stride = torch.tensor([8,16,32]) #需要注释前一行，否则执前时会报错
    def forward(self, x): #就是将", 偏, 放, cen, wh"这些放末
        feat1, feat2, feat3, 偏, 放, cen, wh = self.backbone.forward(x)
        return dbox, cls, x, self.anchors.to(dbox.device), self.strides.to(dbox.device), 偏, 放, cen, wh

#-----------------------------(utils_bbox.py)------------------------#
def dist2bbox1(distance, anchor_points,mim,res,xywh=True,dim=-1):#这插于正牌函后
    lt, rb  = torch.split(distance, 2, dim) #因正牌函在训练时对真实标注转参系后会用到
    x1y1    = anchor_points - lt
    x2y2    = anchor_points + rb#后面是更改的内容，但建议此函数全部替换，因入参有加
    sc=torch.cat((torch.full((6400,),80),torch.full((1600,),40),torch.full((400,),20)),dim=0)
    res = res.unsqueeze(1)
    mim = mim.unsqueeze(1)/640*sc.view(1,-1,1)
    res = res.to(x1y1.device); mim = mim.to(x1y1.device)
    if x1y1.shape[1]==2:res=res.transpose(1,2);mim=mim.transpose(1,2)
    x1y1 = (x1y1-mim)/res#4, 8400, 2
    x2y2 = (x2y2-mim)/res
    if xywh: return torch.cat((((x1y1 + x2y2)/2), x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def decode_box(self, inputs):
    dbox, cls, origin_cls, anchors, strides,mim,res = inputs[:7]
    dbox = dist2bbox1(dbox,anchors.unsqueeze(0),mim,res, xywh=True,dim=1)*strides


#---------------------------(yolo.py)------------------------#
        "model_path"        : 'b.pth',#换为预训的权重，此句后插在原model_path后,加逗

# 训练部分：
#-----------------------------(yolo_training.py)------------------------#
class Loss:
    # gt_labels, gt_bboxes    = targets.split((1, 4), 2)
    mim=preds[-2].to(gt_bboxes.device); res=preds[-1].to(gt_bboxes.device)
    gt_bboxes = gt_bboxes*res.unsqueeze(1).repeat(1,1,2)+mim.unsqueeze(1).repeat(1,1,2)

#-----------------------------(utils_fit.py)------------------------#
import torch.nn.functional as F #补上

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
def fit_one_epoch():
    # val_loss = 0
    cls_loss = 0
    loc_loss = 0#补入两预测损失

    images, bboxes, hms, whs, masks = batch#插入中心预测结果
    with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                bboxes = bboxes.cuda(local_rank)
                hms = hms.cuda(local_rank)
                whs = whs.cuda(local_rank)
                masks = masks.cuda(local_rank)
 else:#进行混合精度fp16训练时
                # outputs         = model_train(images)
                loss_main = yolo_loss(outputs[:7], bboxes)
                loss_cls = focal_loss(outputs[-2], hms)
                loss_loc = reg_l1_loss(outputs[-1], whs, masks)*0.1
                loss_value = loss_main+loss_cls+loss_loc#替原y_loss(outputs,bboxes)
        # loss += loss_main.item()
        cls_loss += loss_cls.item()
        loc_loss += loss_loc.item()
        # if local_rank == 0:
            pbar.set_postfix(**{'loss':loss/(iteration+1),'中':cls_loss/(iteration+1),'尺':loc_loss/(iteration+1),'lr':get_lr(optimizer)})
        #后面验证的时候不需要最后两张预测图所以加[:7]
        outputs     = model_train_eval(images)[:7]
#-----------------------------(dataloader.py)------------------------#
#         if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
#             lines = sample(self.annotation_lines, 3)
#             lines.append(self.annotation_lines[index])
#             shuffle(lines)
#             image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            
#             if self.mixup and self.rand() < self.mixup_prob:
#                 lines           = sample(self.annotation_lines, 1)
#                 image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
#                 image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
#         else:
        image, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train) #去掉马赛克增强，else内容提前
        # box         = np.array(box, dtype=np.float32)
        batch_hm = np.zeros((640,640,1),dtype=np.float32)
        batch_wh = np.zeros((640,640,2),dtype=np.float32)
        batch_mask = np.zeros((640,640),dtype=np.float32)
        for i in range(len(box)):#初始化各图，并随即根据框信息填充，这部分连续复制
            bbox    = box[i].copy()
            cls_id  = int(box[i, -1])
            h = bbox[3]-bbox[1];w = bbox[2]-bbox[0]
            if h > 0 and w > 0: 
                radius = max(0, int((h+w)/6))
                ct=np.array([bbox[0]+w/2,bbox[1]+h/2],dtype=np.float32).astype(np.int32)
                batch_hm[:,:,0] = draw_gaussian(batch_hm[:,:,0], ct, radius)
                batch_wh[ct[1], ct[0]] = 1. * w, 1. * h
                batch_mask[ct[1], ct[0]] = 1
        return image, labels_out, batch_hm, batch_wh, batch_mask

def yolo_dataset_collate(batch): #添加三图的信息，建议取代原函数
    images  = []
    bboxes  = []
    hms,whs,masks     = [],[],[] 
    for i, (img, box, hm, wh, mask) in enumerate(batch):
        images.append(img)
        hms.append(hm)
        box[:, 0] = i
        bboxes.append(box)
        whs.append(wh)
        masks.append(mask)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes  = torch.from_numpy(np.concatenate(bboxes, 0)).type(torch.FloatTensor)
    hms  = torch.from_numpy(np.array(hms)).type(torch.FloatTensor)
    whs  = torch.from_numpy(np.array(whs)).type(torch.FloatTensor)
    masks  = torch.from_numpy(np.array(masks)).type(torch.FloatTensor)
    return images, bboxes, hms,whs,masks

def draw_gaussian(heatmap, center, radius, k=1):#在本py文件末端插入
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h```

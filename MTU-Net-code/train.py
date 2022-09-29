# torch and visulization
from tqdm             import tqdm
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# model
from model.MTU_Net import  res_UNet, Res_block

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 1)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT


        # Choose and load model (this paper is finished by one GPU)

        if args.model == 'MTU_Net':
            model       = res_UNet(num_classes=1, input_channels=args.in_channels, block=Res_block, num_blocks=num_blocks,nb_filter=nb_filter)








        model           = model.cuda()
        model.apply(weights_init_xavier)



        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model, device_ids=[0,1])
        else:
            # model = nn.DataParallel(model)
            model = model.cuda()


        # checkpoint = torch.load('resUnet_IOUloss.tar')
        checkpoint = torch.load('MTU_pretrain.tar')


        # model.load_state_dict(checkpoint['state_dict'],strict=True)

        # checkpoint = torch.load('E:\Infrared-Small-Target-Detection-master-ResU-vitblock01\\mIoU__res_UNet_AsymBi_NUDT-SIRST(sea)_plus_epoch.pth.tar')

        # model.load_state_dict(checkpoint['state_dict'], strict=False)


        print("Model Initializing")
        self.model      = model


        state = self.model.state_dict()
        dict_name = list(state)
        for i, p in enumerate(dict_name):
            print(i, p)

        # Optimizer and lr scheduling
        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()

        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

    def Data_Augmentation(self,args):
        if args.mode == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([ 0.1583,  0.1583, 0.1583], [ 0.0885,  0.0885,  0.0885])])
            transforms.Normalize([ 0.1583,  0.1583, 0.1583], [ 0.0885,  0.0885,  0.0885])])
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True,pin_memory=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False,pin_memory=True)

    # Training
    def training(self,epoch):


        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels) in enumerate(tbar):
            data   = data.cuda()
            labels = labels.cuda()

            pred = self.model(data)

                # loss = SoftIoULoss(pred, labels)
                # loss = WBCELoss(pred, labels)
                # loss = FocalLoss(pred, labels)
            loss_map,loss = FocalIoULoss(pred, labels)
                # loss = QFocalLoss(pred, labels)
                # loss = PDFALoss(pred, labels)

                # loss = SoftIoULoss(pred, labels)
                #loss = IoULoss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        self.ROC.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()

                pred = self.model(data)
                    # loss = FocalLoss(pred, labels)
                loss_map,loss = FocalIoULoss(pred, labels)

                losses.update(loss.item(), pred.size (0))



                self.mIoU.update(pred, labels)


                _, mean_IOU = self.mIoU.get()


                self.PD_FA.update(pred, labels)
                # Final_FA, Final_PD,Final_nIoU = self.PD_FA.get(i+1)
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU ))

            test_loss = losses.avg
            _, mean_IOU = self.mIoU.get()
            # tp_rates, fp_rates, PD, precision, FA  = self.ROC.get()


            FA, PD,nIoU= self.PD_FA.get(847)


            save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                            self.train_loss, test_loss, PD, FA,nIoU[0] , epoch,
                            self.model.state_dict())
            if mean_IOU >self.best_iou:
                self.best_iou = mean_IOU

            self.PD_FA.reset()
            self.ROC.reset()
            self.mIoU.reset()


        # save high-performance model


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.Data_Augmentation(args)
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)






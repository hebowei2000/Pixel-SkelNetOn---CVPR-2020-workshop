import argparse
import sys
from datetime import datetime
import cv2
import imageio

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter

from util import *
from net import *


def train(train_loader, model, optimizer, args):
    # multi resolution
    size_rates = [0.5, 0.75, 1, 1.25, 1.5]

    args.log_path = '../log/' + args.model
    save_path = '../model/' + args.model
    sw = SummaryWriter(args.log_path)

    loss_sal_record, loss_edge_record = AvgMeter(), AvgMeter()
   # loss_dice_record, loss_record = AvgMeter(), AvgMeter()
    total_step = len(train_loader)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    global_step = 0
    test(model, -1, args)
    for epoch in range(0, args.epoch):
        model.train()

        for step, data in enumerate(train_loader, start=1):
            for rate in size_rates:
                optimizer.zero_grad()
                ims, gts, names = data
                # Load data
                ims = ims.cuda()
                gts = gts.cuda()
                # Forward
                trainsize = int(round(args.train_size * rate / 32) * 32)
                if rate != 1:
                    ims = F.upsample(ims, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                    gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                pred_sal = model(ims)

                loss_sal = nn.BCEWithLogitsLoss()(pred_sal, gts)
               # loss_dice=Loss.dice_loss(pred_sal,gts)
               # loss = Loss.structure_loss(pred_sal,gts)
                loss = loss_sal
                loss.backward()

                optimizer.step()
               # log = 'Iteration: {:d} SalLoss: {:.4f} DiceLoss:{:.4f} Loss:{:.4f}'.format(global_step,
                #                                              loss_sal.data.cpu().numpy(),loss_dice.data.cpu().numpy(),loss.data.cpu().numpy())
               # open(args.log_path + '.log', 'a').write(log + '\n')
                if rate == 1:
                    loss_sal_record.update(loss_sal.data, args.batch_size)
                   # loss_dice_record.update(loss_dice.data,args.batch_size)
                    #loss_record.update(loss.data,args.batch_size)

            sw.add_scalar('lr', scheduler.get_lr()[0], global_step=global_step)
            sw.add_scalars('SalLoss', {'SalLoss': loss_sal_record.show()},
                           global_step=global_step)
           # sw.add_scalars('DiceLoss',{'DiceLoss':loss_dice_record.show()},
            #               global_step=global_step)
           # sw.add_scalars('Loss', {'Loss':loss_record.show()},
                            global_step=global_step)
           # log = 'Iteration: {:d} SalLoss: {:.4f} DiceLoss:{:.4f} Loss:{:.4f}'.format(global_step,loss_sal_record.show(),
            #        loss_dice_record.show(),loss_record.show())
            log = 'Iteration: {:d} SalLoss: {:.4f}'.format(global_step,loss_sal_record.show())
            open(args.log_path + '.log','a').write(log + '\n')
            if step % 10 == 0 or step == total_step:
               # print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.6f}, SalLoss: {:.4f}, DiceLoss:{:.4f}, Loss:{:.4f}'.
               #       format(datetime.now(), epoch, args.epoch, step, total_step, scheduler.get_lr()[0],
               #              loss_sal_record.show(),loss_dice_record.show(),loss_record.show()), flush=True)
                 print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR: {:.6f}, SalLoss: {:.4f}'.format(datetime.now(),epoch,args.epoch,
                     step,total_step,scheduler.get_lr()[0],loss_sal_record.show()),flush=True)
            global_step += 1

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if epoch >= 20:
            torch.save(model.state_dict(), save_path + args.model + '_' + '.%d' % epoch + '.pth')

        scheduler.step()

    test(model, -1, args)


def test(model, epoch, args):
    model.eval()
    for dataset in args.valset:
        save_path = './out/' + args.model + '/' + dataset + '/sal/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = args.data_path + dataset + '/'
        gt_root = args.data_path + '/gt/'
        test_loader = SKDataset(image_root, gt_root, args.train_size)
        for i in range(test_loader.size):
            image, name = test_loader.load_data()
            image = image.cuda()
            attention = model(image)
            attention = F.upsample(attention, size=(256, 256), mode='bilinear', align_corners=True)
            res = attention.sigmoid().data.cpu().numpy().squeeze()
            # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            ret,res_bi=cv2.threshold(res,7,255,cv2.THRESH_BINARY)
            imageio.imsave('../data/result/' + name + '.png', res_bi)


def main():
    # init parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--train_size', type=int, default=256, help='training dataset size')
    parser.add_argument('--trainset', type=str, default='DUTS_TRAIN', help='training  dataset')
    parser.add_argument('--channel', type=int, default=30, help='channel number of convolutional layers in decoder')
    parser.add_argument('--is_resnet', type=bool, default=True, help='VGG or ResNet backbone')
    parser.add_argument('--model', type=str, default='baseline', help='VGG or ResNet backbone')
    args = parser.parse_args()

    np.random.seed(2020)
    torch.manual_seed(2020)
    torch.cuda.manual_seed(2020)

    print('Learning Rate: {} ResNet: {} Trainset: {}'.format(args.lr, args.is_resnet, args.trainset))

    # build model
    model = globals()[args.model]()
    model.cuda()

    params = model.parameters()
    optimizer = torch.optim.SGD(params, args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(params, args.lr, weight_decay=5e-4)

    # dataset
    args.data_path = '../data/'
    image_root = args.data_path + 'train/'
    gt_root = args.data_path + 'gt/'
    train_loader = sk_loader(image_root, gt_root, args.batch_size, args.train_size)
    args.valset = ['test']

    # begin training
    print("Time to witness the mirracle!")
    train(train_loader, model, optimizer, args)


if __name__ == '__main__':
    main()

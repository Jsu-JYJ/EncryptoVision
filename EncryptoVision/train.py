import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from my_dataset import MyDataSet
from model import vit_base as create_model
from utils import read_split_data, train_one_epoch, evaluate


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    train_dataset = MyDataSet(images_path=train_images_path, images_class=train_images_label, transform=None,
                              data_type=args.dataset)
    val_dataset = MyDataSet(images_path=val_images_path, images_class=val_images_label, transform=None,
                            data_type=args.dataset)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 16])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes, has_logits=False).to(device)
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)

        print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            if "head" not in name and "pre_logits" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=5E-5)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc, precision, recall, f1score, w1, w2 = evaluate(model=model,
                                                                         data_loader=val_loader,
                                                                         device=device,
                                                                         epoch=epoch,
                                                                         args=args)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "precision", "recall", "f1score", "w1", "w2"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], precision, epoch)
        tb_writer.add_scalar(tags[5], recall, epoch)
        tb_writer.add_scalar(tags[6], f1score, epoch)
        tb_writer.add_scalar(tags[7], w1, epoch)
        tb_writer.add_scalar(tags[8], w2, epoch)
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch + 1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--data-path', type=str, default="", help='dataset/ISCX-VPN2016/Train')
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or cpu)')
    opt = parser.parse_args()
    main(opt)

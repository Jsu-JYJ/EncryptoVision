import argparse
from torch.utils.data import DataLoader

from my_dataset import MyDataSet
from utils import *
from model import vit_base as create_model


def main(args):
    model = create_model(num_classes=args.num_classes, has_logits=False).to(args.device)

    test_img, test_label = load_test_data(root=args.data_path)

    test_dataset = MyDataSet(images_path=test_img, images_class=test_label, data_type=args.dataset)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             num_workers=args.num_workers, shuffle=False, pin_memory=True, drop_last=True)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=args.device)
        print(model.load_state_dict(weights_dict, strict=False))

    val_loss, val_acc, val_pre, val_recall, val_f1, _, _ = evaluate(model, test_loader, args.device, 0, args)

    print(f'[Metrics]  loss:{val_loss:.4f} acc:{val_acc:.4f} pre:{val_pre:.4f} recall:{val_recall:.4f} F1-score:{val_f1:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='')
    parser.add_argument('--data-path', type=str, default="", help='dataset/ISCX-VPN2016/Test')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or cpu)')
    opt = parser.parse_args()
    main(opt)
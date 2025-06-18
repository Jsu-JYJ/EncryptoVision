import os
import sys
import json
import random
import numpy
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    traffic_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    traffic_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(traffic_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in traffic_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label


def load_test_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    traffic_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    traffic_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(traffic_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    test_images_path = []
    test_images_label = []
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in traffic_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        images.sort()
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        for img_path in images:
            test_images_path.append(img_path)
            test_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    assert len(test_images_path) > 0, "number of training images must greater than 0."

    return test_images_path, test_images_label


def confusion_matrix(preds, labels, conf_matrix):
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return conf_matrix


def draw_conf_matrix(conf_matrix, num, epoch):
    # labels = ['Attacks_Flood', 'Attacks_Hydra', 'Attacks_Nmap', 'Idle','Interactions_Audio',
    # 'Interactions_Cameras', 'Interactions_Other', 'Power_Audio', 'Power_Cameras', 'Power_Other']

    labels = ['BROWSING', 'CHAT', 'MAIL', 'FTP','P2P', 'Streaming', 'VOIP']
    # labels = ['Advertising', 'Analytics & Telemetry', 'Antivirus', 'Authentication services','Blogs & News',
    # 'default', 'E-commerce',  'File sharing', 'Games', 'Information systems', 'Instant messaging', 'Mail', 'Music',
    # 'Other services and APIs', 'Search', 'Social', 'Streaming media', 'Videoconferencing']

    # labels = ['BitTorrent', 'Cridex', 'Facetime', 'FTP', 'Geodo', 'Gmail', 'Htbot', 'Miuref', 'MySQL', 'Neris', 'Nsis-ay',
    #           'Outlook', 'Shifu', 'Skype', 'SMB', 'Tinba', 'Virut', 'Weibo', 'WorldOfWarcraft', 'Zeus']
    # 显示数据
    plt.figure(figsize=(8, 8), dpi=100)
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    plt.colorbar()
    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 1.5

    for x in range(num):
        for y in range(num):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     fontsize=8,
                     color="white" if info > thresh else "black")

    plt.tight_layout()
    plt.yticks(range(num), labels)
    plt.xticks(range(num), labels, rotation=90)
    plt.subplots_adjust(left=0.16, right=1, bottom=0.25, top=0.9, wspace=1.0, hspace=1.0)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig('confusion/VPN-{}.png'.format(epoch + 1))


def get_evaluate_data(conf_matrix):
    TP = conf_matrix.diagonal()
    FP = conf_matrix.sum(1) - TP
    FN = conf_matrix.sum(0) - TP

    # macro average
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    precision = precision.mean()
    recall = recall.mean()

    f1score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1score


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, sequence, labels = data
        sample_num += images.shape[0]
        pred, _ = model(images.to(device), sequence.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch, args):
    loss_function = torch.nn.CrossEntropyLoss()
    model.eval()
    conf_matrix = torch.zeros(args.num_classes, args.num_classes).to(device)
    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, sequence, labels = data
        sample_num += images.shape[0]

        pred, weights = model(images.to(device),sequence.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss.detach()
        conf_matrix = confusion_matrix(pred_classes, labels, conf_matrix)

        data_loader.desc = "[Valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                        accu_loss.item() / (step + 1),
                                                        accu_num.item() / sample_num)
    conf_matrix = numpy.array(conf_matrix.cpu())
    precision, recall, f1score = get_evaluate_data(conf_matrix)
    # if (epoch + 1) % 4 == 0:
    # draw_conf_matrix(conf_matrix, args.num_classes, epoch)
    weight1 = torch.mean(weights[:, 0])
    weight2 = torch.mean(weights[:, 1])

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num, precision, recall, f1score, weight1, weight2

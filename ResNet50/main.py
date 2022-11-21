import torch
from torch import nn, optim
from model import resnet50
from densenet import DenseNet_BC
from dataloaders import *


def main():
    Batch_size = 64
    epoch = 100000
    num_workers = 8
    best_acc = 0.000
    best_epoch = -1
    init_lr = 1e-4 # 初始学习率Initial learning rate
    lambda1 = lambda epoch: epoch // 30  # 第一组参数的调整方法Method of adjusting the first set of parameters
    lambda2 = lambda epoch: 0.95 ** epoch  # 第二组参数的调整方法The second set of parameter adjustment methods
    train_path = "./datasets/train"
    test_path = "./datasets/val"

    trainset = foot_dataset(train_path)
    trainloader = DataLoader(dataset=trainset, batch_size=Batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    testset = foot_dataset(test_path)
    testloader = DataLoader(dataset=testset, batch_size=Batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = resnet50().to(device)
    # model = DenseNet_BC().to(device)

    criteon = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1, lambda2]) # 动态调整学习率Dynamically adjust the learning rate
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[200, 400, 600, 800, 1000], gamma=0.85)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    for epoch in range(epoch):
        model.train()
        for batchidx, (x, label) in enumerate(trainloader):
            x, label = x.to(device), label.to(device)
            logits,_ = model(x)
            loss = criteon(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Epoch: %4s, step: %3s, loss: %.10f, best acc: %.3f, best_epoch: %4s" %(epoch, batchidx, loss.item(), best_acc, best_epoch))

        if epoch % 10 == 0 and epoch != 0:
            model.eval()
            with torch.no_grad():
                # test
                print("Begin testing...")
                total_correct = 0
                total_num = 0
                for x, label in testloader:
                    x, label = x.to(device), label.to(device)

                    logits, _ = model(x)
                    pred = logits.argmax(dim=1)

                    correct = torch.eq(pred, label).float().sum().item()
                    total_correct += correct
                    total_num += x.size(0)

                acc = total_correct / total_num
                print("Current test acc: %s" % acc)
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch
                    best_model_path = "./model"
                    if not os.path.exists(best_model_path):
                        os.makedirs(best_model_path)
                    torch.save(model, os.path.join(best_model_path, "best.pth"))
        # 动态更新学习率Update the learning rate dynamically
        scheduler.step()


if __name__ == '__main__':
    main()

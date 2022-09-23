import torch

from torch.optim import Adam
from torch_geometric.data import DataLoader
from sklearn import metrics


def train_multiple_epochs(train_dataset, test_dataset, model, args):

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=2)

    test_size = 1024  # load all test dataset
    test_loader = DataLoader(test_dataset, test_size, shuffle=False,
                             num_workers=2)

    model.to(args.device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    pbar = range(start_epoch, args.epochs + start_epoch)

    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, args.device)

        if epoch % args.valid_interval == 0:
            roc_auc, aupr = evaluate_metric(model, test_loader, args.device)

            print("epoch {}".format(epoch), "train_loss {0:.4f}".format(train_loss),
                  "roc_auc {0:.4f}".format(roc_auc), "aupr {0:.4f}".format(aupr))


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()
    total_loss = 0
    pbar = loader
    for data in pbar:
        optimizer.zero_grad()
        true_label = data.to(device)
        predict = model(true_label)
        loss_function = torch.nn.BCEWithLogitsLoss()
        loss = loss_function(predict, true_label.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()

    return total_loss / len(loader.dataset)


def evaluate_metric(model, loader, device):
    model.eval()
    pbar = loader
    roc_auc, aupr = None, None
    for data in pbar:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)

        y_true = data.y.view(-1).cpu().tolist()
        y_score = out.cpu().numpy().tolist()

        fpr, tpr, threshold = metrics.roc_curve(y_true, y_score)
        roc_auc = metrics.auc(fpr, tpr)

        aupr = metrics.average_precision_score(y_true, y_score)
        torch.cuda.empty_cache()

    return roc_auc, aupr
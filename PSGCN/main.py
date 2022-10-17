import argparse

from train_eval import *
from models import *
from util_functions import load_k_fold


def extract_subgraph(split_data_dict, k):
    if args.data_name == 'Gdataset':
        print("Using Gdataset with 10% testing...")
        (
            adj_train, train_labels, train_u_indices, train_v_indices,
            test_labels, test_u_indices, test_v_indices
        ) = split_data_dict[k]

    elif args.data_name == 'Cdataset':

        print("Using Cdataset with 10% testing...")
        (
            adj_train, train_labels, train_u_indices, train_v_indices,
            test_labels, test_u_indices, test_v_indices
        ) = split_data_dict[k]  # load_drug_data(path)

    else:
        print("Using LRSSL with 10% testing...")
        (
            adj_train, train_labels, train_u_indices, train_v_indices,
            test_labels, test_u_indices, test_v_indices
        ) = split_data_dict[k]

    val_test_appendix = str(k) + '_kfold'
    data_combo = (args.data_name, val_test_appendix)

    train_indices = (train_u_indices, train_v_indices)
    test_indices = (test_u_indices, test_v_indices)

    train_file_path = 'data/{}/{}/train'.format(*data_combo)
    train_graph = MyDataset(train_file_path, adj_train, train_indices, train_labels, args.hop)
    # train_graph = MyDynamicDataset(train_file_path, adj_train, train_indices, train_labels, args.hop)

    test_file_path = 'data/{}/{}/test'.format(*data_combo)
    test_graph = MyDataset(test_file_path, adj_train, test_indices, test_labels, args.hop)
    # test_graph = MyDynamicDataset(test_file_path, adj_train, test_indices, test_labels, args.hop)

    return train_graph, test_graph


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description='PSGCN')

    parser.add_argument('--data-name', default='Gdataset',
                        help='dataset name')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--hop', type=int, default=2,
                        help='the number of neighbor (default: 2)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=30, metavar='E',
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='batch size during training')
    parser.add_argument('--dropout', type=float, default=0.4,
                        help='random drops neural node and edge with this prob')
    parser.add_argument('--force-undirected', action='store_true', default=False,
                        help='in edge dropout, force (x, y) and (y, x) to be dropped together')
    parser.add_argument('--save_dir', type=str, help='The saving directory')
    parser.add_argument('--valid_interval', type=int, default=1)

    args = parser.parse_args()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    print(args)

    seeds = [12, 34, 42, 43, 61, 70, 83, 1024, 2014, 2047]

    for seed in seeds:
        print("=============", str(seed), "==================")
        split_data_dict = load_k_fold(args.data_name, seed)
        
        for k in range(0, 10):
            args.save_dir = args.data_name + "_" + ''.join(str(k+1) + 'fold')
            args.save_dir = os.path.join("result", args.save_dir)
            print('=================fold', str(k + 1), '=================')
            train_graphs, test_graphs = extract_subgraph(split_data_dict, k)
            model = PSGCN(
                train_graphs,
                latent_dim=[64, 64, 1],
                k=10,
                dropout=args.dropout,
                force_undirected=args.force_undirected
            )

            print('Used #train graphs: %d, #test graphs: %d' % (
                len(train_graphs),
                len(test_graphs),
            ))

            train_multiple_epochs(train_graphs, test_graphs, model, args)
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    
    # federated learning arguments
    parser.add_argument('--method', type=str, default='krum', help="aggregation method")
    parser.add_argument('--global_ep', type=int, default=200, help="total number of communication rounds")
    parser.add_argument('--alpha', default=10.0, type=float, help="random distribution fraction alpha")
    parser.add_argument('--num_clients', type=int, default=10, help="number of clients: K")
    parser.add_argument('--num_data', type=int, default=100, help="number of data per client for label skew")
    parser.add_argument('--quantity_skew', action='store_true', help='quantity_skew')
    parser.add_argument('--num_pretrain', type=int, default=50, help="number of data for pretraining")
    parser.add_argument('--frac', type=float, default=1.0, help="fraction of clients: C")
    parser.add_argument('--ratio', type=float, default=1.0, help="ratio of datasize")
    parser.add_argument('--server_ep', type=int, default=10, help="number of epochs for pretraining")
    parser.add_argument('--local_ep', type=int, default=5, help="number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=20, help="test batch size")
    parser.add_argument('--ds', type=int, default=20, help="dummy batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="client learning rate")

    # other arguments
    parser.add_argument('--dataset', type=str, default='bloodmnist', help="name of dataset")
    parser.add_argument('--model', type=str, default='resnet', help='model name')
    parser.add_argument('--sampling', type=str, default='noniid', help="sampling method")
    parser.add_argument('--sampling_classes', type=int, default=5, help="number of classes for sampling")
    parser.add_argument('--num_classes', type=int, default=8, help="number of classes")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=3, help='random seed (default: 1)')
    parser.add_argument('--tsboard', action='store_true', help='tensorboard')
    parser.add_argument('--debug', action='store_true', help='debug')
    parser.add_argument('--earlystop', action='store_true', help='early stopping option')
    parser.add_argument('--patience', type=int, default=8, help="hyperparameter of early stopping")
    parser.add_argument('--delta', type=float, default=0.01, help="hyperparameter of early stopping")

    # poisoning arguments
    parser.add_argument('--c_frac', default=0.0, type=float, help="fraction of compromised clients")
    parser.add_argument('--mp_alpha', type=float, default=10.0, help="hyperparameter for targeted model attack")
    parser.add_argument('--p', type=str, default='normal', help="model poisoning attack (target, untarget) or data poisoning")
    parser.add_argument('--mp_lambda', type=float, default=10.0, help="hyperparameter for untargeted model attack")

    args = parser.parse_args()
    
    return args
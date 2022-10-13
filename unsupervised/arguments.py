import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='GcnInformax Arguments.')
    parser.add_argument('--DS', dest='DS', help='Dataset')
    parser.add_argument('--rate1', type=float, default=0.1)
    parser.add_argument('--rate2', type=float, default=0.25)
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, 
            help='Learning rate.')
    parser.add_argument('--num-gc-layers', dest='num_gc_layers', type=int, default=5,
            help='Number of graph convolution layers before each pooling')
    parser.add_argument('--hidden-dim', dest='hidden_dim', type=int, default=32,
            help='')
    
    parser.add_argument('--lambda1', type=float, default=1.0, help='weight of FI')
    parser.add_argument('--lambda2', type=float, default=0.01, help='weight of BT')
    parser.add_argument('--lambda3', type=float, default=0.01, help='weight of HSIC')
    parser.add_argument('--aug', type=str, default='dnodes')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    
    return parser.parse_args()


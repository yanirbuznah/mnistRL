import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--save_dir",
                    type=str,
                    default='./pretrained_mnist_guesser_models',
                    help="Directory for saved models")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=512,
                    help="Hidden dimension")
parser.add_argument("--lr",
                    type=float,
                    default=1e-4,
                    help="Learning rate")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.,
                    help="l_2 weight penalty")
parser.add_argument("--case",
                    type=int,
                    default=2,
                    help="Which data to use")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=50,
                    help="Number of validation trials without improvement")
parser.add_argument("--val_interval",
                    type=int,
                    default=1000,
                    help="Interval for calculating validation reward and saving model")


FLAGS = parser.parse_args(args=[])
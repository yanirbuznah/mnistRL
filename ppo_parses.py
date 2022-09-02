import argparse

parser = argparse.ArgumentParser(description="PPO args bidpedalHardcore")
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--learning_cycles', type=int, default=2048, help='Number learning cycles')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs per learning cycles')
parser.add_argument('--buffer_size', type=int, default=2048,
                    help='Number of samples to generate on each learning cycle')
parser.add_argument('--lr', type=float, default=0.00025, help='Learning rate')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--policy_clip', type=float, default=0.1, help='Policy clip')
# parser.add_argument('--update_factor',type = int, default = 2048, help = 'Update factor')
parser.add_argument('--gae_lambda', type=float, default=0.99, help='Lambda')
parser.add_argument('--std', type=float, default=0.8, help='initial std')

parser.add_argument("--ep_per_trainee",
                    type=int,
                    default=1000,
                    help="Switch between training dqn and guesser every this # of episodes")
parser.add_argument("--max-episode",
                    type=int,
                    default=20000,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min-eps",
                    type=float,
                    default=0.1,
                    help="Min epsilon")
parser.add_argument("--save_dir",
                    type=str,
                    default='./mnist_ddqn_models',
                    help="Directory for saved models")
parser.add_argument("--masked_images_dir",
                    type=str,
                    default='./mnist_masked_images',
                    help="Directory for saved masked images")
parser.add_argument("--min_lr",
                    type=float,
                    default=1e-5,
                    help="Minimal learning rate")
parser.add_argument("--decay_step_size",
                    type=int,
                    default=50000,
                    help="LR decay step size")
parser.add_argument("--lr_decay_factor",
                    type=float,
                    default=0.1,
                    help="LR decay factor")
parser.add_argument("--weight_decay",
                    type=float,
                    default=0.0,
                    help="l_2 weight penalty")
parser.add_argument("--val_interval",
                    type=int,
                    default=1000,
                    help="Interval for calculating validation reward and saving model")
parser.add_argument("--episode_length",
                    type=int,
                    default=6,
                    help="Episode length")
parser.add_argument("--case",
                    type=int,
                    default=2,
                    help="Which data to use")
parser.add_argument("--env",
                    type=str,
                    default="mnist",
                    help="environment name: Questionnaire")
# Environment params
parser.add_argument("--g_hidden-dim",
                    type=int,
                    default=512,
                    help="Guesser hidden dimension")
parser.add_argument("--g_weight_decay",
                    type=float,
                    default=0e-4,
                    help="Guesser l_2 weight penalty")







parser.add_argument('-f')

FLAGS = parser.parse_args(args=[])

import argparse

# parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument("--save_dir",
#                     type=str,
#                     default='./mnist_ddqn_models',
#                     help="Directory for saved models")
# parser.add_argument("--masked_images_dir",
#                     type=str,
#                     default='./mnist_masked_images',
#                     help="Directory for saved masked images")
# parser.add_argument("--gamma",
#                     type=float,
#                     default=0.85,
#                     help="Discount rate for Q_target")
# parser.add_argument("--n_update_target_dqn",
#                     type=int,
#                     default=10,
#                     help="Mumber of episodes between updates of target dqn")
# parser.add_argument("--val_trials_wo_im",
#                     type=int,
#                     default=200,
#                     help="Number of validation trials without improvement")
# parser.add_argument("--ep_per_trainee",
#                     type=int,
#                     default=1000,
#                     help="Switch between training dqn and guesser every this # of episodes")
# parser.add_argument("--batch_size",
#                     type=int,
#                     default=64,
#                     help="Mini-batch size")
# parser.add_argument("--hidden-dim",
#                     type=int,
#                     default=64,
#                     help="Hidden dimension")
# parser.add_argument("--capacity",
#                     type=int,
#                     default=10000,
#                     help="Replay memory capacity")
# parser.add_argument("--max-episode",
#                     type=int,
#                     default=10000,
#                     help="e-Greedy target episode (eps will be the lowest at this episode)")
# parser.add_argument("--min-eps",
#                     type=float,
#                     default=0.1,
#                     help="Min epsilon")
# parser.add_argument("--lr",
#                     type=float,
#                     default=1e-4,
#                     help="Learning rate")
# parser.add_argument("--min_lr",
#                     type=float,
#                     default=1e-5,
#                     help="Minimal learning rate")
# parser.add_argument("--decay_step_size",
#                     type=int,
#                     default=50000,
#                     help="LR decay step size")
# parser.add_argument("--lr_decay_factor",
#                     type=float,
#                     default=0.1,
#                     help="LR decay factor")
# parser.add_argument("--weight_decay",
#                     type=float,
#                     default=0e-4,
#                     help="l_2 weight penalty")
# parser.add_argument("--val_interval",
#                     type=int,
#                     default=1000,
#                     help="Interval for calculating validation reward and saving model")
# parser.add_argument("--episode_length",
#                     type=int,
#                     default=6,
#                     help="Episode length")
# parser.add_argument("--case",
#                     type=int,
#                     default=2,
#                     help="Which data to use")
# parser.add_argument("--env",
#                     type=str,
#                     default="Questionnaire",
#                     help="environment name: Questionnaire")
# # Environment params
# parser.add_argument("--g_hidden-dim",
#                     type=int,
#                     default=512,
#                     help="Guesser hidden dimension")
# parser.add_argument("--g_weight_decay",
#                     type=float,
#                     default=0e-4,
#                     help="Guesser l_2 weight penalty")
#
#




parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--save_dir",
                    type=str,
                    default='./mnist_ddqn_models',
                    help="Directory for saved models")
parser.add_argument("--masked_images_dir",
                    type=str,
                    default='./mnist_masked_images',
                    help="Directory for saved masked images")
parser.add_argument("--gamma",
                    type=float,
                    default=0.99,
                    help="Discount rate for Q_target")
parser.add_argument("--n_update_target_dqn",
                    type=int,
                    default=10,
                    help="Mumber of episodes between updates of target dqn")
parser.add_argument("--val_trials_wo_im",
                    type=int,
                    default=200,
                    help="Number of validation trials without improvement")
parser.add_argument("--ep_per_trainee",
                    type=int,
                    default=1000,
                    help="Switch between training dqn and guesser every this # of episodes")
parser.add_argument("--batch_size",
                    type=int,
                    default=16,
                    help="Mini-batch size")
parser.add_argument("--hidden-dim",
                    type=int,
                    default=256,
                    help="Hidden dimension")
parser.add_argument("--capacity",
                    type=int,
                    default=10000,
                    help="Replay memory capacity")
parser.add_argument("--max-episode",
                    type=int,
                    default=200000,
                    help="e-Greedy target episode (eps will be the lowest at this episode)")
parser.add_argument("--min-eps",
                    type=float,
                    default=0.1,
                    help="Min epsilon")
parser.add_argument("--lr",
                    type=float,
                    default=5e-4,
                    help="Learning rate")
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
                    default=10,
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
                    default=128,
                    help="Guesser hidden dimension")
parser.add_argument("--g_weight_decay",
                    type=float,
                    default=0e-4,
                    help="Guesser l_2 weight penalty")


FLAGS = parser.parse_args(args=[])
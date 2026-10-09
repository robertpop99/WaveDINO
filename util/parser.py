import argparse
from multiprocessing import cpu_count


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='WaveDINO: atmospheric noise removal for unwrapped InSAR')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log', action='store_false', default=True,
                        help='log the run')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-frequency', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-frequency', type=int, default=50, metavar='N',
                        help='how many epochs to wait before saving the model (default: 50)')
    parser.add_argument('--test-frequency', type=int, default=5, metavar='N',
                        help='how often to test the model on the test dataset')
    parser.add_argument('--dataset', type=str, default='taal',
                        help='train and test dataset')
    parser.add_argument('--datasets-dir', type=str, default='datasets',
                        help='train and test dataset')
    parser.add_argument('--num-channels', type=int, default=1, metavar='N',
                        help='number of channels (default: 1)')
    parser.add_argument('--result-dir', type=str, default='results',
                        help='directory where to save model weights and results')
    parser.add_argument('--metadata', type=str, default='',
                        help='Information to add to the metadata file')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='directory where to save the logs')
    parser.add_argument("-n_jobs", "--num-workers", default=cpu_count(), type=int,
                        help="Number of worker processes used to load data.")
    parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="Learning rate")
    parser.add_argument("--alpha", default=1., type=float, help='Alpha for SWIN')
    parser.add_argument('--model-name', type=str,
                        help='name of the model to be loaded')
    parser.add_argument('--evaluate', '--validate', action='store_true', default=False, help='Run test only')
    parser.add_argument('--no-patches', action='store_true', default=False,
                        help='With --evaluate: skip the full-frame sliding-window inference, run resize only')
    parser.add_argument('--synthetic-test', action='store_true', default=False,
                        help='With --evaluate: run the synthetic test split instead of the real-data GNSS evaluation')
    parser.add_argument('--test-limit', type=int, default=5620,
                        help='With --synthetic-test: number of test samples')
    parser.add_argument('--load-checkpoint', action='store_true', default=False,
                        help='Load a model checkpoint and continue training')
    # parser.add_argument('--decoder_type', type=str, default='gaussian', help='save output images')
    parser.add_argument('--image-size', type=int, default=128, metavar='N', help='size of the images')
    parser.add_argument('--model-type', type=str, default='na', help='model to use for training')
    parser.add_argument('--no-heatmap', action='store_true', default=False, help='disables heatmap during testing')
    parser.add_argument('--wrap', action='store_true', default=False, help='The dataset has wrapped images')
    parser.add_argument('--downsample', action='store_true', default=False, help='Downsample the images before use')
    parser.add_argument('--debug', action='store_true', default=False, help='Debug mode')
    parser.add_argument('--interpolation', type=str, default='new', help='interpolation type')
    return parser.parse_args(args)

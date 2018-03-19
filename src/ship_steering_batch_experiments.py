from ship_steering_server import server_experiment
from ship_steering_server_small import server_experiment_small
import argparse
import datetime
from mushroom.utils.folder import *

def batch_experiments():

    how_many = 20
    parser = argparse.ArgumentParser(description='server_harch_ship')
    parser.add_argument("--small", help="environment size small or big", action="store_true")
    args = parser.parse_args()
    small = args.small
    print('SMALL IS', small)


    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'

    for i in range(how_many):
        print('experiment number:  ', i)
        if small:
            server_experiment_small(i, subdir)
        else:
            server_experiment(i, subdir)

    force_symlink(subdir, 'latest')


if __name__ == '__main__':
    batch_experiments()

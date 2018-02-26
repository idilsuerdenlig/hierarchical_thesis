from server_harch_ship import server_experiment
import argparse
import datetime
from mushroom.utils.folder import *

def batch_experiments():

    how_many = 50

    parser = argparse.ArgumentParser(description='server_harch_ship')
    parser.add_argument("--small", help="environment size small or big", action="store_true")
    args = parser.parse_args()
    small = args.small
    print 'SMALL IS', small

    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'

    for i in xrange(how_many):
        print 'experiment number:  ', i
        server_experiment(small, i, subdir)

    force_symlink(subdir, 'latest')


if __name__ == '__main__':
    batch_experiments()

from ship_steering_server import server_experiment
import argparse
import datetime
from mushroom.utils.folder import *

def batch_experiments():

    how_many = 20


    subdir = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '/'

    for i in range(how_many):
        print('experiment number:  ', i)
        server_experiment(i, subdir)

    force_symlink(subdir, 'latest')


if __name__ == '__main__':
    batch_experiments()

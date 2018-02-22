from server_harch_ship import server_experiment
from visualize_saved_in_server import visualize_saved_in_server
import argparse
from mushroom.utils.folder import *

def batch_experiments():

    how_many = 50

    parser = argparse.ArgumentParser(description='server_harch_ship')
    parser.add_argument("--small", help="environment size small or big", action="store_true")
    args = parser.parse_args()
    small = args.small
    small = True
    print 'SMALL IS', small

    for i in xrange(how_many):
        server_experiment(small, i)
        print 'experiment number:  ', i

if __name__ == '__main__':
    batch_experiments()

from model.template import *
import random

def flood_run():
    random.seed(1)
    flood([8,4,1], [1,2,1], 0.01, 0.01, 2000)
    flood([8,4,1], [1,2,1], 0.03, 0.05, 2000)
    flood([8,8,1], [1,2,1], 0.01, 0.01, 2000)
    flood([8,2,2,1], [1,2,2,1], 0.01, 0.01, 2000)

def cross_run():
    random.seed(1)
    cross([2,4,2], [1,4,1], 0.01, 0.01, 2500)
    cross([2,4,2], [1,4,1], 0.06, 0.005, 2500)
    cross([2,8,2], [1,4,1], 0.01, 0.01, 2500)
    cross([2,4,4,2], [1,4,4,1], 0.01, 0.01, 2500)

if __name__ == '__main__':
    flood_run()
    # cross_run()
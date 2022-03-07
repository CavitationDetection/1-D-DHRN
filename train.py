from main import train
from opts import parse_opts
from utils import *


# train and test our model
if __name__ == "__main__":
    opts = parse_opts()
    create_dirs('./figs')
    create_dirs('./outputs')
    create_dirs('./models')
    train(opts)

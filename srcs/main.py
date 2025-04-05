import os, sys
import traceback
import time
import argparse
import yaml
import numpy as np
import torch
sys.path.append('/home/lmh/projects_dir/graduate/RememberThePast-DatasetDistillation')
from utils import (
    command_parser,
    dict2namespace,
    merge_args,
)

from runners.sgld_runner_standard import *


def parse_args_and_config():
    # parse arguments
    parser = command_parser()
    args   = parser.parse_args()
    # args.debug = True
    if args.device :
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "7"
        
    if args.debug:
        args.bptt_inner_steps = 5
        
    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
        config = dict2namespace(config)

    # merge arguments and config
    config = merge_args(args, config)

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, config
    

def main():
    # args and config
    # os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    args, config = parse_args_and_config()
    
    print(f"using device {os.environ['CUDA_VISIBLE_DEVICES']}")
        
    setattr(config, 'pid', str(os.getpid()))

    # basic info
    print("Exp instance id = {}".format(config.pid))
    print("Exp note = {}".format(args.note))
    print("Config = {}".format(args.config))
    print(">" * 80)
    print(config)
    print("<" * 80)

    try:
        train_multi_run(config, args)
    except:
        print(traceback.format_exc())

    return 0

if __name__ == '__main__':
    sys.exit(main())

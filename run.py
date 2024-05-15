import deli
import argparse


def print_results(res):
    print('LOSSES STATS')
    print(f'mean = {res[0][0]}')
    print(f'std = {res[0][1]}')
    print('NUMBER OF PARAMETERS')
    print(f'res[1]')
    print('MODEL')
    print(res[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    # parser.add_argument('seed', type=int, default=0)
    # parser.add_argument('input_dim', type=int, default=5)
    # parser.add_argument('num_workers', type=int, default=8)
    # parser.add_argument('batch_size', type=int, default=512)
    # parser.add_argument('num_threads', type=int, default=1)
    # parser.add_argument('hidden_dim', type=int, default=20)
    # parser.add_argument('lr', type=float, default=1e-2)
    # parser.add_argument('num_epochs', type=int, default=20)
    # parser.add_argument('num_launches', type=int, default=4)
    # parser.add_argument('num_epochs', type=int, default=1e-20)

    args = parser.parse_args()
    config = deli.load_json(args.config)

    DEVICE = args.device
    SEED = config["SEED"]
    INPUT_DIM = config["INPUT_DIM"]
    NUM_WORKERS = config["NUM_WORKERS"]
    BATCH_SIZE = config["BATCH_SIZE"]
    NUM_THREADS = config["NUM_THREADS"]
    HIDDEN_DIM = config["HIDDEN_DIM"]
    LR = config["LR"]
    NUM_EPOCHS = config["NUM_EPOCHS"]
    NUM_LAUNCHES = config["NUM_LAUNCHES"]
    GAIN = config["GAIN"]
    SPACE_SIZE = config["SPACE_SIZE"]


    from src.train import (train_classic_mlp, 
                       train_unpruned_mlp,
                       train_pruned_mlp)
    
    
    print('------------ START TRAIN CLASSIC MLP ------------')
    classic_results = train_classic_mlp(**config)
    print('------------ END TRAIN CLASSIC MLP ------------')
    print_results(classic_results)

    print('------------ START TRAIN UNPRUNED MLP ------------')
    unpruned_results = train_unpruned_mlp(**config)
    print('------------ END TRAIN UNPRUNED MLP ------------')
    print_results(unpruned_results)

    print('------------ START TRAIN PRUNED MLP ------------')
    pruned_results = train_pruned_mlp(*config, num_batch=1)
    print('------------ END TRAIN PRUNED MLP ------------')
    print_results(pruned_results)


    

                       


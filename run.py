import deli
import argparse


def print_results(res):
    print('LOSSES STATS')
    print(f'mean = {res[0][0]}')
    print(f'std = {res[0][1]}')
    print('NUMBER OF PARAMETERS')
    print(res[1])
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

    config['device'] = args.device
    # SEED = config["seed"]
    # INPUT_DIM = config["input_dim"]
    # NUM_WORKERS = config["num_workers"]
    # BATCH_SIZE = config["batch_size"]
    # NUM_THREADS = config["num_threads"]
    # HIDDEN_DIM = config["hidden_dim"]
    # LR = config["lr"]
    # NUM_EPOCHS = config["num_epochs"]
    # NUM_LAUNCHES = config["num_launches"]
    # GAIN = config["gain"]
    # SPACE_SIZE = config["space_size"]


    from src.train import (train_classic_mlp, 
                       train_unpruned_mlp,
                       train_pruned_mlp)
    
    
    print('------------ START TRAIN CLASSIC MLP ------------')
    classic_results = train_classic_mlp(**config)
    print_results(classic_results)
    print('------------ END TRAIN CLASSIC MLP ------------')

    print('------------ START TRAIN UNPRUNED MLP ------------')
    unpruned_results = train_unpruned_mlp(**config)
    print_results(unpruned_results)
    print('------------ END TRAIN UNPRUNED MLP ------------')

    for num_batch in range(1, 6):
        print(f'------------ START TRAIN PRUNED MLP num_batch = {num_batch} ------------')
        pruned_results = train_pruned_mlp(**config, num_batch=num_batch)
        print_results(pruned_results)
        print('------------ END TRAIN PRUNED MLP ------------')


    

                       


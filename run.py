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

    args = parser.parse_args()
    config = deli.load_json(args.config)

    config['device'] = args.device



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


    

                       


from argparse import ArgumentParser
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

all_accs = []

def run_current_experiment(task_type, config, seed=0):
    global accuracies
    # Run current experiment
    run_experiment_command = 'python control-tasks/run_experiment.py %s --seed=%d' % (config, seed)
    os.system(run_experiment_command)

    # Extract current accuracy
    output_folder = 'results/%s_%d/' % (task_type, seed)
    acc_path = os.path.join(output_folder, 'dev.label_acc')

    with open(acc_path) as acc_file:
        acc = float(acc_file.readline().strip()[:4]) # restrict to 2 decimal points
        all_accs.append(acc)
    
    # Delete current output folder
    clear_storage_command = 'rm -r %s' % (output_folder)
    os.system(clear_storage_command)

    return acc

if __name__ == "__main__":
    argp = ArgumentParser()
    argp.add_argument('--dataset', choices=['en_ewt', 'ptb'],
        help='choose which dataset to use')
    argp.add_argument('--num-probe-layers', type=int, choices=[0, 1],
        help='specify desired probe complexity')
    argp.add_argument('--num-tests', type=int,
      help='total number of tests to be run')
    cli_args = argp.parse_args()

    linguistic_config = "example/config/%s-pos-corrupted0-rank1000-%dhid-ELMo1.yaml" % (cli_args.dataset, cli_args.num_probe_layers)
    control_config = "example/config/%s-pos-corrupted1-rank1000-%dhid-ELMo1.yaml" % (cli_args.dataset, cli_args.num_probe_layers)

    # Run linguistic test and all control tests
    linguistic_acc = run_current_experiment('linguistic', linguistic_config)
    accs = [linguistic_acc]
    for i in range(cli_args.num_tests-1):
        curr_control_acc = run_current_experiment('control', control_config, seed=i)
        accs.append(curr_control_acc)

    # Plot histogram with linguistic accuracy bin highlighted in red
    print("Recorded Accuracies:\n", accs)
    import pdb
    pdb.set_trace()
    nums, bins, patches = plt.hist(all_accs, bins=np.arange(0, 1.01, 0.01), color='g')
    linguistic_idx = np.where(bins == linguistic_acc)[0][0]
    patches[linguistic_idx].set_fc('r')
    plt.savefig('results/histogram.png')

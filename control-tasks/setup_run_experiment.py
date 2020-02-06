from argparse import ArgumentParser
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

linguistic_linear_config = "example/config/en_ewt-pos-corrupted0-rank1000-0hid-ELMo1.yaml"
control_linear_config = "example/config/en_ewt-pos-corrupted1-rank1000-0hid-ELMo1.yaml"
linguistic_mlp_config = "example/config/en_ewt-pos-corrupted0-rank1000-1hid-ELMo1.yaml"
control_mlp_config = "example/config/en_ewt-pos-corrupted1-rank1000-1hid-ELMo1.yaml"

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
    argp.add_argument('--num-probe-layers', type=int, choices=[0, 1],
        help='specify desired probe complexity')
    argp.add_argument('--num-tests', type=int,
      help='total number of tests to be run')
    cli_args = argp.parse_args()

    if cli_args.num_probe_layers == 0:
        linguistic_config = linguistic_linear_config
        control_config = control_linear_config
    else:
        assert cli_args.num_probe_layers == 1
        linguistic_config = linguistic_mlp_config
        control_config = control_mlp_config

    # Run linguistic test and all control tests
    linguistic_acc = run_current_experiment('linguistic', linguistic_config)
    accs = [linguistic_acc]
    for i in range(cli_args.num_tests-1):
        curr_control_acc = run_current_experiment('control', control_config, seed=i)
        accs.append(curr_control_acc)

    # Plot histogram with linguistic accuracy bin highlighted in red
    print("Recorded Accuracies:\n", accs)
    nums, bins, patches = plt.hist(all_accs, bins=np.arange(0, 1.01, 0.01), color='g')
    linguistic_idx = np.where(bins == linguistic_acc)[0][0]
    patches[linguistic_idx].set_fc('r')
    plt.savefig('results/histogram.png')

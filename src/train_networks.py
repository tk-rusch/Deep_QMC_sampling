import os
from sys import argv

if __name__ == '__main__':
    experiment_type = argv[1]
    train_type = argv[2]
    max_iterations = argv[3]

    if(train_type == 'ensemble'):
        comline = 'bsub -J "calc[1-1008]" "python train.py \$LSB_JOBINDEX ' + \
                  experiment_type + ' '  + train_type + ' ' + max_iterations + '" '
    else:
        comline = 'bsub -J "calc[1-700]" "python train.py \$LSB_JOBINDEX ' + \
                  experiment_type + ' '  + train_type + ' ' + max_iterations + '" '

    os.system(comline)


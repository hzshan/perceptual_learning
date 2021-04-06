#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p sompolinsky_gpu
#SBATCH --gres=gpu:1
#SBATCH --gpu-freq=high



module load python
module load cuda/9.0-fasrc02
source activate $ENV_NAME
cd /n/home11/haozheshan/$FOLDER_NAME
python $SCRIPT_NAME  \
--BATCH_NAME $BATCH_NAME \
--TRIAL_IND $TRIAL_IND \
--cluster 1 \
--N $N \
--Nhid $Nhid \
--nonlinearity $nonlinearity \
--n_layers $n_layers \
--noise_var $noise_var \
--eta $eta \
--loss $loss \
--n_train_trial $n_train_trials \
--test_interval $test_interval \
--sig_w $sig_w \
--sig_s $sig_s \
--lambda2 $lambda2 \
--n_learn $n_learn

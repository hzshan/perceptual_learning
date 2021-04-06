BATCH_NAME='3L_MSE_GD_lambda_sigs0p2_sigw0p8_logscale_'
# BATCH_NAME='3L_MSE_GD_sigw_sigsP2_new__'
# Naming convention (Aug 24, 2020)
# Number-of-layers_loss-type_dynamics_varied-parameter_addition-parameter__

SCRIPT_NAME='main_no_converge.py'
# SCRIPT_NAME='main.py'

SBATCH_NAME='main.sbatch' # main.sbatch // main_own_partition.sbatch // main_fas.sbatch
FOLDER_NAME='PL'
ENV_NAME='continual'

directory="/n/home11/haozheshan/${FOLDER_NAME}/Raw_results/${BATCH_NAME}/"
rm -rf $directory  # remove the directory 
mkdir $directory
``
### Amount of resources to request from the cluster for each job
MEMORY_REQUESTED=2000  # in MB
TIME_REQUESTED=0-6:00  # format: D-HH:MMhahahah
 
nonlinearity='relu'
loss='MSE'
eta=0.001 # default 0.001
n_train_trials=50
test_interval=100000
n_layers=3
noise_var=0.01
N=1000
Nhid=1000
sig_s=0.2
sig_w=0.8
# lambda2=.02 # check this
lambda2=.01
n_learn=40000000

export SCRIPT_NAME BATCH_NAME SBATCH_NAME FOLDER_NAME ENV_NAME nonlinearity eta n_train_trials loss test_interval noise_var sig_w n_learn

# values=(0 0.05 0.1 0.15 0.2)
# values=$(seq 0.1 0.05 0.6)
values_lambda=(0.0001 0.00032 0.001 0.0032 0.01 0.032 0.1 0.32 1 3.2)

TRIAL_IND=1

for lambda2 in ${values_lambda[@]}
do

  for avg_ind in $(seq 1 1 1)
  do

    TRIAL_IND_str=$( printf '%03d' $TRIAL_IND)

    export N Nhid TRIAL_IND n_layers sig_s lambda2
    sbatch -o /n/home11/haozheshan/${FOLDER_NAME}/Raw_results/${BATCH_NAME}/Output_${TRIAL_IND_str}.txt \
    --job-name=$BATCH_NAME \
    --time=$TIME_REQUESTED \
    --mem $MEMORY_REQUESTED \
    /n/home11/haozheshan/${FOLDER_NAME}/${SBATCH_NAME}
    echo $lambda2

    TRIAL_IND=$(($TRIAL_IND+1))
    sleep 0.5
  done
done
echo $BATCH_NAME

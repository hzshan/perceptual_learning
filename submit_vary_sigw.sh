# BATCH_NAME='6L_MSE_GD_sigw_sigs0p2_etaP005_'
BATCH_NAME='3L_MSE_GD_sigw_sigsP2__'
# Naming convention (Aug 24, 2020)
# Number-of-layers_loss-type_dynamics_varied-parameter_addition-parameter__

# SCRIPT_NAME='main_no_converge.py'
SCRIPT_NAME='main.py'

SBATCH_NAME='main_fas.sbatch' # main.sbatch // main_own_partition.sbatch // main_fas.sbatch
FOLDER_NAME='PL'
ENV_NAME='continual'

directory="/n/home11/haozheshan/${FOLDER_NAME}/Raw_results/${BATCH_NAME}/"
rm -rf $directory  # remove the directory 
mkdir $directory

### Amount of resources to request from the cluster for each job
MEMORY_REQUESTED=2000  # in MB
TIME_REQUESTED=0-4:00  # format: D-HH:MMhahahah
 
nonlinearity='relu'
loss='MSE'
eta=0.001 #default 0.001
n_train_trials=50  #default 50
test_interval=50000
n_layers=3
noise_var=0.01
N=1000
Nhid=1000
sig_s=0.2
# lambda2=.02 # check this
lambda2=.0
n_learn=20000000
# If N=2000, use n_learn=6000000

export SCRIPT_NAME BATCH_NAME SBATCH_NAME FOLDER_NAME ENV_NAME nonlinearity eta n_train_trials loss test_interval noise_var lambda2 n_learn

# values=(0 0.05 0.1 0.15 0.2)
# values=$(seq 0.1 0.05 0.6)
values_sigw=$(seq 0.1 0.05 1.0)

TRIAL_IND=1

for sig_w in ${values_sigw[@]}
do

  for avg_ind in $(seq 1 1 1)
  do

    TRIAL_IND_str=$( printf '%03d' $TRIAL_IND)

    export N Nhid TRIAL_IND n_layers sig_w sig_s
    sbatch -o /n/home11/haozheshan/${FOLDER_NAME}/Raw_results/${BATCH_NAME}/Output_${TRIAL_IND_str}.txt \
    --job-name=$BATCH_NAME \
    --time=$TIME_REQUESTED \
    --mem $MEMORY_REQUESTED \
    --account=cox_lab \
    /n/home11/haozheshan/${FOLDER_NAME}/${SBATCH_NAME}
    echo $sig_w

    TRIAL_IND=$(($TRIAL_IND+1))
    sleep 0.5
  done
done
echo $BATCH_NAME

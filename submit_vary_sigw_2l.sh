BATCH_NAME='2Layers_SquaredError_GradDescent_sigw_sigs0p2_'
# always add a "_" in the end, if the last characters of the batch name belong to a 
# parameter value e.g. "xxxx_N100_" instead of "xxxx_N100".
SCRIPT_NAME='main.py'
SBATCH_NAME='main_gpu.sbatch'
FOLDER_NAME='PL'
ENV_NAME='continual'

directory="/n/home11/haozheshan/${FOLDER_NAME}/Raw_results/${BATCH_NAME}/"
rm -rf $directory  # remove the directory 
mkdir $directory

### Amount of resources to request from the cluster for each job
MEMORY_REQUESTED=2000  # in MB
TIME_REQUESTED=0-3:00  # format: D-HH:MMhahahah
 
nonlinearity='relu'
loss='MSE'
eta=0.001
n_train_trials=50
test_interval=50000
n_layers=2
noise_var=0.01
N=1000
Nhid=1000
sig_s=0.2
lambda2=.00 # check this
n_learn=2000000

export SCRIPT_NAME BATCH_NAME SBATCH_NAME FOLDER_NAME ENV_NAME nonlinearity eta n_train_trials loss test_interval noise_var lambda2 n_learn

# values=(0 0.05 0.1 0.15 0.2)
# values=$(seq 0.1 0.05 0.6)
values=$(seq 0.1 0.1 1.0)

TRIAL_IND=1

for sig_w in ${values[@]}
do

  for avg_ind in $(seq 1 1 1)
  do

    TRIAL_IND_str=$( printf '%03d' $TRIAL_IND)

    export N Nhid TRIAL_IND n_layers sig_w sig_s
    sbatch -o /n/home11/haozheshan/${FOLDER_NAME}/Raw_results/${BATCH_NAME}/Output_${TRIAL_IND_str}.txt \
    --job-name=$BATCH_NAME \
    --time=$TIME_REQUESTED \
    --mem $MEMORY_REQUESTED \
    /n/home11/haozheshan/${FOLDER_NAME}/${SBATCH_NAME}
    echo $sig_w

    TRIAL_IND=$(($TRIAL_IND+1))
    sleep 0.5
  done
done
echo $BATCH_NAME

BATCH_NAME='3L_MSE_GD_sigsP2_sigwP8_etap01_reps'

SCRIPT_NAME='main.py' # main.py or main_new_ro.py
SBATCH_NAME='main_fas.sbatch' # main.sbatch // main_own_partition.sbatch // main_fas.sbatch
FOLDER_NAME='PL'
ENV_NAME='continual'

### Amount of resources to request from the cluster for each job
MEMORY_REQUESTED=4000  # in MB
TIME_REQUESTED=0-1:00  # format: D-HH:MMhahahah

directory="/n/home11/haozheshan/${FOLDER_NAME}/Raw_results/${BATCH_NAME}/"
rm -rf $directory  # remove the directory 
mkdir $directory

nonlinearity='relu'
loss='MSE'
eta=0.01
n_train_trials=25
test_interval=100000
n_layers=3
noise_var=0.01
sig_w=0.8
sig_s=0.2
lambda2=0.0
n_learn=8000000

export SCRIPT_NAME BATCH_NAME SBATCH_NAME FOLDER_NAME ENV_NAME nonlinearity eta n_train_trials loss test_interval noise_var sig_w sig_s lambda2 n_learn

values=(1000)
# values=(400 800 1600 3200)

TRIAL_IND=1


for N in ${values[@]}
do
  for avg_ind in $(seq 1 1 20)
  do

    Nhid=$N
    TRIAL_IND_str=$( printf '%03d' $TRIAL_IND)

    export N Nhid TRIAL_IND n_layers
    sbatch -o /n/home11/haozheshan/${FOLDER_NAME}/Raw_results/${BATCH_NAME}/Output_${TRIAL_IND_str}.txt \
    --job-name=$BATCH_NAME \
    --time=$TIME_REQUESTED \
    --mem $MEMORY_REQUESTED \
    /n/home11/haozheshan/${FOLDER_NAME}/${SBATCH_NAME}


    TRIAL_IND=$(($TRIAL_IND+1))
    sleep 0.5
  done
done
echo $BATCH_NAME

#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH -c 16
#SBATCH --job-name="pred_l20"
#SBATCH --output=chr2chr3_l.txt
#SBATCH --partition=general

# centers="5"
# seed=10
# path=2cohorts
# srun python -m cloudpred data/signature/1.0_0.5_110_1/ -t log --logfile log/signature_1.0_0.5_110_1 \
# --genpat --deepset --centers 5 --dims 25 --valid 50 --test 50 --pc True

# srun python -m cloudpred 1cohort/chr1_50/ -t none --logfile log/1cohort/chr1_50 \
# --cloudpred --linear --generative --genpat --deepset --centers ${centers} --valid 0.25 --test 0.25 --figroot fig/chr1_50/

# srun python -m cloudpred ../../$path/chr2chr1_$seed/ -t none -d 20 --pc False --seed $seed --logfile ../../log/$path/chr2chr1_$seed \
# --cloudpred --linear --generative --genpat --deepset  --centers ${centers} --valid 0.0 --test 0.0 --figroot ../../fig/$path_$seed/


# srun python -m cloudpred data/signature/1.0_0.5_130_1/ -t log --logfile log/signature_1.0_0.5_110_1 \
# --cloudpred --linear --generative --genpat --deepset --centers 5 --dims 25 --valid 50 --test 50 --pc True


# cp -r ../../2cohorts/chr2chr1 ../../2cohorts/chr2chr1_0
# cp -r ../../2cohorts/chr2chr1 ../../2cohorts/chr2chr1_0

# cp -r ../../2cohorts/chr2chr1 ../../2cohorts/chr2chr1_0
# cp -r ../../2cohorts/chr2chr1 ../../2cohorts/chr2chr1_0


srun python prediction.py
# srun python gmm_class.py
# srun python gmm_patient.py
# srun python prediction_chr2chr1_cloudPred.py

# Run scripts/compute_acuity_scores.py to compute additional acuity scores with the raw patient features.
# 
# Run scripts/split_sepsis_cohort.py to compose a training, validation and testing split as well as remove binary or demographic information from the temporal patient features. This script also organizes the patient data into convenient trajectory formats for easier use with sequential or recurrent models.
# 
# Use Behavior Cloning on the data provided by the previous step to develop a baseline "expert" policy for use in training and evaluating RL policies from the learned patient representations.
# This is done by running scripts/train_behavCloning.py. An example of how this can be done is provided in slurm_scripts/slurm_build_BC.py --> slurm_scripts/slurm_bc_exp.
# This will generate either behav_policy_file or behav_policy_file_wDemo which is used internal to Steps 2 and 3 below.

# https://github.com/MLforHealth/rl_representations/blob/main/scripts/split_sepsis_cohort.py

'''
This script preprocesses and organizes the Sepsis patient cohort extracted with the procedure 
provided at: https://github.com/microsoft/mimic_sepsis to produce patient trajectories for easier
use in sequential models.
============================================================================================================================
This code is provided under the MIT License and is meant to be helpful, but WITHOUT ANY WARRANTY;
November 2020 by Taylor Killian and Haoran Zhang; University of Toronto + Vector Institute
============================================================================================================================
Notes:
'''

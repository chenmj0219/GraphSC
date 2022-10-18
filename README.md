# MGSC

This is the code of MGSC!

# Dependencies

·torch==1.10.2 <br>
·torch-geometric==2.0.3 <br>
·torch-scatter==2.0.9 <br>
·torch-sparse==0.6.12 <br>
·scikit-learn-intelex <br>
·tqdm <br>
·networkx==2.5.1 <br>

# Dataset 
  ## unsupervised
  https://chrsmrrs.github.io/datasets/docs/datasets/ to download dataset.<br>
  ## transfer
  https://github.com/snap-stanford/pretrain-gnns#dataset-download to download dataset. <br>
# Training & Evaluation
  ## Unsupervised
      cd unsupervised
      sh run.sh

  ## Transfer
   ### Pre-train
      cd transfer
      sh pretrain.sh 
   
   ### Fine-tune
      sh finetune.sh
   
# Hyperparameters
  The settings of the hyperparameters are detailed in the experimental section and the appendix section of the paper.
  
# Special Matters
  In an unsupervised learning settings, you should replace the OneHotDegree function in the aug.py file with the corresponding file in our directory, which is only used to get the feature matrix for the social network dataset.




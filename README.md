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
  cd unsupervised <br>
  sh run.sh <br>

  ## Transfer
   ### Pre-train
      cd transfer <br>
      sh pretrain.sh <br> 
   
   ### Fine-tune
      sh finetune.sh <br>
   
# Hyperparameters
  The settings of the hyperparameters are detailed in the experimental section and the appendix section of the paper.


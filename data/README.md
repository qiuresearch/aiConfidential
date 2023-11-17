# Contents
Analyses of SeqFold2D models developed on Strive 2022 dataset with 50% random splits between training and validation sets

* Each model is a folder with the prefix "strive.libset_len30-600_nr80_train-valid.l4c64.scheduler_validobj__rand50" (total 75 models)
* All data are from libset_len30-600_nr80_train-valid.pkl, 50% of which were used for training
* Performance on the training data is saved in the file "strive_2022.libset_len30-600_nr80_train-valid_TR.eval/eval_loss_meta.csv", with each sample identified as "idx" and the evalation metric as "f1".
* Performance on the rest is saved in the file "strive_2022.libset_len30-600_nr80_train-valid_VL.eval/eval_loss_meta.csv",
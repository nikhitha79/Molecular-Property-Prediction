Code for the model training is stored in final.py. Optuna training code is stored as dpm_optuna.py and corresponding sql db as db.sqlite3. The ASE db for the dataset is named as new_dataset70.db. Predicted values are stored in predicted200.csv. The trained model is inside the new_dataset70 folder names as best_inference_model_fold. A report for the model is attached in the PDF format. 

Additionally, we also calculated 2D and 3D descriptors relevant to the dipole moment and built various models to find a correlation to predict dipole moment accurately. These descriptors show poor correlation as dipole moment is a complex property which requires vector representations. The descriptors are stored in the file calculated_descriptors.csv.

References

1. SchNetPack: A Deep Learning Toolbox For Atomistic Systems, K. T. Schütt, P. Kessel, M. Gastegger, K. A. Nicoli, A. Tkatchenko, and K.-R. Müller, Journal of Chemical Theory and Computation 2019 15 (1), 448-455, DOI: 10.1021/acs.jctc.8b00908
   
2. K. T. Schütt, O. T. Unke, and M. Gastegger, Equivariant message passing for the prediction of tensorial properties and molecular spectra, arXiv preprint arXiv:2102.03150, 2021. https://arxiv.org/abs/2102.03150

3. T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, Optuna: A Next-generation Hyperparameter Optimization Framework, Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 2019, pp. 2623–2631. https://doi.org/10.1145/3292500.3330701

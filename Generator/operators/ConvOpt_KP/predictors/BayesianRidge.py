from sklearn.linear_model import BayesianRidge
from MyDataLoader import HLS_Dataloader, Abs_Diff
import numpy as np
import pickle


train_ratio = 0.6
val_ratio = 0.2


if __name__ == '__main__':
	HLS_Data = HLS_Dataloader(train_ratio=train_ratio, val_ratio=val_ratio)
	X_train, Y_train, X_val, Y_val, X_test, Y_test = HLS_Data.get_datasets()

	model_dsp = BayesianRidge()
	model_dsp.fit(X_train, Y_train[:, 1])

	pkl_filename = "BRR_dsp.pkl"
	model_save = {'model': model_dsp, 'mean': HLS_Data.Y_mean[1], 'std': HLS_Data.Y_std[1]}
	with open(pkl_filename, 'wb') as file:
	    pickle.dump(model_save, file)

	# Load from file
	with open(pkl_filename, 'rb') as file:
	    pickle_model = pickle.load(file)
	mean = pickle_model['mean']
	std = pickle_model['std']
	pred = pickle_model['model']
	# Calculate the accuracy score and predict target values
	y_hat = pred.predict(X_test)

	print(f"Test MAE: {Abs_Diff(y_hat, Y_test[:, 1]) * HLS_Data.Y_std[1]}")
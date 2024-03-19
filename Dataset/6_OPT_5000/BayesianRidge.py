import argparse
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from MyDataLoader import HLS_Dataloader, MAE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle
import pathlib
from xgboost import XGBRegressor

test_size = 0.2


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--data-file', help='file name of operator samples')
	parser.add_argument('-p', '--packing', help='Packing methods for the operator', choices=['FP', 'KP'])
	parser.add_argument('--LUT', action='store_true')
	parser.add_argument('--DW', action='store_true')
	opt = parser.parse_args()
	dataset_path = opt.data_file
	Packing = opt.packing
	LUT = opt.LUT
	DW = opt.DW
	
	HLS_Data = HLS_Dataloader(dataset_path, test_size, Packing, LUT, DW)
	X_train, Y_train, X_test, Y_test = HLS_Data.get_datasets()

	if HLS_Data.samp_num > 0:
		print(f'Found {HLS_Data.samp_num} samples in total (Packing: {Packing}, LUT: {LUT}, DW: {DW})!')
		opt_path = pathlib.Path(f'./{Packing}_{DW}_{LUT}/')
		if not opt_path.is_dir():
			opt_path.mkdir()
	else:
		raise ValueError(f'Error! Found no samples (Packing: {Packing}, LUT: {LUT}, DW: {DW}).')

	targets = ['DSP', 'LUT', 'BRAM', 'WNS']


	for idx, tar in enumerate(targets):
		# BRR = make_pipeline(StandardScaler(), BayesianRidge())
		BRR = make_pipeline(StandardScaler(), XGBRegressor(learning_rate=0.1, max_depth=8, subsample=0.8, reg_lambda=8.0, n_estimators=200))
		BRR.fit(X_train, Y_train[:, idx])
	
		pkl_filename = opt_path / f"BRR_{tar}.pkl"
		with open(pkl_filename, 'wb') as file:
		    pickle.dump(BRR, file)
	
		# Load from file
		with open(pkl_filename, 'rb') as file:
		    pickle_model = pickle.load(file)
		# y_hat = pickle_model.predict(X_test)
		y_hat = np.maximum(pickle_model.predict(X_test), 0.0)
	
		print(f"Test MAE: {MAE(y_hat, Y_test[:, idx])}, targets: {tar}")


	# # II
	# HLS_Data = HLS_Dataloader(dataset_path, test_size, Packing, LUT, DW, True)
	# X_train, Y_train, X_test, Y_test = HLS_Data.get_datasets()
	# print(f'Found {HLS_Data.samp_num} samples in total (Packing: {Packing}, LUT: {LUT}, DW: {DW})!')

	# BRR = make_pipeline(StandardScaler(), LogisticRegression())
	# BRR.fit(X_train, Y_train[:, 4])
	
	# pkl_filename = opt_path / f"BRR_II.pkl"
	# with open(pkl_filename, 'wb') as file:
	#     pickle.dump(BRR, file)
	
	# # Load from file
	# with open(pkl_filename, 'rb') as file:
	#     pickle_model = pickle.load(file)
	
	# print(f"Test Acc: {pickle_model.score(X_test, Y_test[:, 4])}, targets: II")


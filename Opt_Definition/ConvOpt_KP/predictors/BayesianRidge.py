from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LogisticRegression
from MyDataLoader import HLS_Dataloader, MAE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


test_size = 0.1


if __name__ == '__main__':
	# WNS, DSP, LUT, BRAM
	HLS_Data = HLS_Dataloader(test_size)
	X_train, Y_train, X_test, Y_test = HLS_Data.get_datasets()

	targets = ['wns', 'dsp', 'lut', 'bram']

	for idx, tar in enumerate(targets):
		BRR = make_pipeline(StandardScaler(), BayesianRidge())
		BRR.fit(X_train, Y_train[:, idx])
	
		pkl_filename = f"BRR_{tar}.pkl"
		with open(pkl_filename, 'wb') as file:
		    pickle.dump(BRR, file)
	
		# Load from file
		with open(pkl_filename, 'rb') as file:
		    pickle_model = pickle.load(file)
		y_hat = pickle_model.predict(X_test)
	
		print(f"Test MAE: {MAE(y_hat, Y_test[:, idx])}, targets: {tar}")

	# II
	HLS_Data = HLS_Dataloader(test_size, True)
	X_train, Y_train, X_test, Y_test = HLS_Data.get_datasets()

	BRR = make_pipeline(StandardScaler(), LogisticRegression())
	BRR.fit(X_train, Y_train[:, 4])
	
	pkl_filename = f"BRR_II.pkl"
	with open(pkl_filename, 'wb') as file:
	    pickle.dump(BRR, file)
	
	# Load from file
	with open(pkl_filename, 'rb') as file:
	    pickle_model = pickle.load(file)
	
	print(f"Test Acc: {pickle_model.score(X_test, Y_test[:, 4])}, targets: II")
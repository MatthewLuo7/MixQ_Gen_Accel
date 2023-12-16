from xgboost import XGBRegressor
from MyDataLoader import HLS_Dataloader, MAE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle


test_size = 0.1


if __name__ == '__main__':
	HLS_Data = HLS_Dataloader(test_size=test_size)
	X_train, Y_train, X_test, Y_test = HLS_Data.get_datasets()

	BRR_dsp = make_pipeline(StandardScaler(), XGBRegressor(learning_rate=0.1, max_depth=8, subsample=0.9, reg_lambda=8.4, n_estimators=191))
	BRR_dsp.fit(X_train, Y_train[:, 1])

	pkl_filename = "XGB_dsp.pkl"
	with open(pkl_filename, 'wb') as file:
	    pickle.dump(BRR_dsp, file)

	# Load from file
	with open(pkl_filename, 'rb') as file:
	    pickle_model = pickle.load(file)
	y_hat = pickle_model.predict(X_test)

	print(f"Test MAE: {MAE(y_hat, Y_test[:, 1])}")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import numpy.random as nr
import math
import os
from datetime import datetime
from sklearn.linear_model import LinearRegression, SGDRegressor
import sys
import time
import imp
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import train_test_split
import lightgbm as lgb






def drop_duplicate(data, sub_set):
    print('Before drop shape:', data.shape)
    before = data.shape[0]
    data.drop_duplicates(sub_set, keep='first', inplace=True)
    data.reset_index(drop=True, inplace=True)
    print('After drop shape:', data.shape)
    after = data.shape[0]
    print('Total Duplicate:', before - after)

def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


class predict(object):

	def __init__(self,trainfile,testfile):
		self.trainfile = trainfile
		self.testfile = testfile
		self.__lr = LinearRegression()
		# self.__dtree = DecisionTreeClassifier()
		# self.__rforest = RandomForestClassifier()
		# self.__svm = SVC(kernel='rbf')
		self.lgb_params = {
        'feature_fraction': 1,
        'metric': 'rmse',
        'min_data_in_leaf': 16,
        'bagging_fraction': 0.85,
        'learning_rate': 0.03,
        'objective': 'mse',
        'bagging_seed': 2 ** 7,
        'num_leaves': 32,
        'bagging_freq': 3,
        'verbose': 0
    	}
		self.__tree_reg = ExtraTreesRegressor(n_estimators=600, max_depth=38,random_state=50)
		self._xgb = XGBRegressor(max_depth=8,n_estimators=1000,min_child_weight=300,colsample_bytree=0.9,subsample=0.9,eta=0.15,seed=42)
		self.train_data = None
		self.train_labels = None
		self.train_data1 = None
		self.train_labels1 = None
		self.val_data = None
		self.val_labels = None
		self.test_data = None
		self.predicted_labels = None
		self.x_train_val = None
		self.y_train_val = None

	def trainingdata(self):
		parser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')
		df = pd.read_csv(self.trainfile,parse_dates=['date'],date_parser=parser)
		df = df.dropna()
		df = df.loc[df['item_cnt_day']>0]
		subset_train = ['date', 'date_block_num', 'shop_id', 'item_id', 'item_cnt_day']
		drop_duplicate(df, sub_set=subset_train)
		median = df[(df.shop_id == 32) & (df.item_id == 2973) & (df.date_block_num == 4) & (df.item_price > 0)].item_price.median()
		df.loc[df.item_price < 0, 'item_price'] = median
		df['item_cnt_day'] = df['item_cnt_day'].clip(0, 1000)
		df['item_price'] = df['item_price'].clip(0, 300000)
		df.loc[df.shop_id == 0, 'shop_id'] = 57
		df.loc[df.shop_id == 1, 'shop_id'] = 58
		df.loc[df.shop_id == 10, 'shop_id'] = 11
	
		df['day'] = df['date'].apply(lambda x: x.strftime('%d'))
		df['day'] = df['day'].astype('int64')
		df['month'] = df['date'].apply(lambda x: x.strftime('%m'))
		df['month'] = df['month'].astype('int64')
		df['year'] = df['date'].apply(lambda x: x.strftime('%Y'))
		df['year'] = df['year'].astype('int64')
		df = df[['day','month','year','item_id', 'shop_id','item_price','item_cnt_day']]
		df['item_id'] = np.log1p(df['item_id'])
		self.train_labels1 = df['item_cnt_day']
		self.train_data1 = df.drop(columns='item_cnt_day')
		self.train_data,self.val_data,self.train_labels,self.val_labels=train_test_split(self.train_data1,self.train_labels1,test_size=0.3)
		self.x_train_val = self.train_data[-100:]
		self.y_train_val = self.train_labels[-100:]


	def testingdata(self):
		parser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')
		df = pd.read_csv(self.testfile,parse_dates=['date'],date_parser=parser)
		subset_test = ['date', 'date_block_num', 'shop_id', 'item_id']
		drop_duplicate(df, sub_set=subset_test)
		df.loc[df.shop_id == 0, 'shop_id'] = 57
		df.loc[df.shop_id == 1, 'shop_id'] = 58
		df.loc[df.shop_id == 10, 'shop_id'] = 11
		df['day'] = df['date'].apply(lambda x: x.strftime('%d'))
		df['day'] = df['day'].astype('int64')
		df['month'] = df['date'].apply(lambda x: x.strftime('%m'))
		df['month'] = df['month'].astype('int64')
		df['year'] = df['date'].apply(lambda x: x.strftime('%Y'))
		df['year'] = df['year'].astype('int64')
		df = df[['day','month','year','item_id', 'shop_id','item_price']]
		df['item_id'] = np.log1p(df['item_id'])
		self.test_data = df;

	def data(self):
		self.trainingdata()
		self.testingdata()

	def trainLinearRegression(self):
		self.__lr.fit(self.train_data,self.train_labels)

	def testLinearRegression(self):
		self.predicted_labels =  self.__lr.predict(self.val_data)
		# print ("Linear Regression score " + str(self.__lr.score(self.val_data, self.val_labels)))
		print ("Linear Regression score " + str(rmse(self.predicted_labels,self.val_labels)))

	def trainExtraTreeRegressor(self):
		self.__tree_reg.fit(self.train_data,self.train_labels)

	def testExtraTreeRegressor(self):
		self.predicted_labels =  self.__tree_reg.predict(self.val_data)
		print ("ExtraTreeRegressor score " + str(rmse(self.predicted_labels,self.val_labels)))

	def trainLightGBM(self):
		lgb.train(self.lgb_params,lgb.dataset(self.train_data,label=train_labels),300)

	def testLightGBM(self):
		self.predicted_labels =  lgb.predict(self.val_data)
		print ("LightGBM  score " + str(rmse(self.predicted_labels,self.val_labels)))

	def trainXGBoost(self):
		self.__xgb.fit(self.train_data,self.train_labels,eval_metric="rmse",eval_set=[(self.train_data, self.train_labels), (self.x_train_val, self.y_train_val)],verbose=True,early_stopping_rounds=10)

	def testXGBoost(self):
		self.predicted_labels = self.__xgb.predict(self.val_data)
		print ("XGBoost  score " + str(rmse(self.predicted_labels,self.val_labels)))







if __name__ == "__main__":
	train_data_name = sys.argv[1]
	test_data_name = sys.argv[2]
	model = predict(train_data_name,test_data_name)
	model.data()
	# model.trainLinearRegression()
	# model.testLinearRegression()

	# model.trainExtraTreeRegressor()
	# model.testExtraTreeRegressor()

	# model.trainLightGBM()
	# model.testLightGBM()

	# model.trainXGBoost()
	# model.testXGBoost()


	# plotConfusionMatrix(model.test_labels,model.predicted_labels)
	
	# model.trainDecesionTree()
	# model.testDecesionTree()

	# model.trainRandomForrest()
	# model.testRandomForrest()

	# model.trainSVM()
	# model.testSVM()






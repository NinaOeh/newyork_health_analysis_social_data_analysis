"""
Created on Thu May 5, 2022

@author: MS
"""

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from random import randint
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
import scipy
from plotly.io import write_image
import folium
from streamlit_folium import folium_static
import pandas as pd
import numpy as np

import geopandas as gpd
import folium
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import json
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from random import randint
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
import scipy
from plotly.io import write_image

import bokeh
from bokeh.plotting import figure,output_file, show
from bokeh.models import ColumnDataSource, Legend, HoverTool, LabelSet
from bokeh.io import output_notebook, reset_output
from bokeh.layouts import layout
from bokeh.models.widgets import Tabs, Panel
from bokeh.io import curdoc
from bokeh.models import FactorRange
from bokeh.models import ColumnDataSource, Grid, HBar, LinearAxis, Plot
from bokeh.plotting import figure, output_file, save
from bokeh.models import Span
from IPython.display import IFrame
from IPython.core.display import display, HTML
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

import xgboost as xgb

def loading_df():
	# Loading Air quality dataset
	df_air = pd.read_csv("./data/air_quality/Air_Quality.csv")
	df_air['Start_Date'] = pd.to_datetime(df_air['Start_Date'])
	df_air['Year'] = df_air['Start_Date'].dt.to_period('Y').astype('str')
	df_air = df_air[df_air['Year']=='2015']
	df_air = df_air[['Name', 'Measure', 'Measure Info', 'Geo Join ID', 'Geo Place Name', 'Data Value']]
	df_air_col_1 = df_air[df_air['Name']=='Fine Particulate Matter (PM2.5)']
	df_air_col_2 = df_air[df_air['Name']=='Sulfur Dioxide (SO2)']
	df_air_col_1 = df_air_col_1.groupby('Geo Join ID').mean()
	df_air_col_1.columns = ['Fine Particulate Matter (PM2.5)'] 
	df_air_col_2 = df_air_col_2.groupby('Geo Join ID').mean()
	df_air_col_2.columns = ['Sulfur Dioxide (SO2)']

	# Helper dictionary
	id_neighbourhood_dict = {}
	for id_num in set(df_air['Geo Join ID']):
		id_neighbourhood_dict[id_num] = df_air[df_air['Geo Join ID'] == id_num]['Geo Place Name'].iloc[0]

	# Final Air Quality Dataset
	df_air = pd.concat([df_air_col_1, df_air_col_2], axis=1, join="inner")
	
	# Loading Health dataset
	health_data = pd.read_csv('./data/health/2015_Health2.csv',sep=";")
	# Load the health dataset and filter only for the two columns of interest
	health_data.drop(labels=[0,1,2,3,4,5],axis=0,inplace=True)
	health_data=health_data.reset_index()
	focus_health=["ID","Avoidable_Asthma","Life_expectancy_rate", "Poverty","Smoking","Unemployment","Self_rep_health"]
	health_data=health_data.loc[:, health_data.columns.isin(focus_health)]
	health_data=health_data.iloc[:59]
	health_data=health_data.set_index('ID')
	health_data.index = health_data.index.astype('int64')
	
	# Loading Trees dataset
	df_trees_count = pd.read_csv("./data/trees/areas_tree_counts.csv")

	#Index by the district for the tree counts and areas
	df_trees_count = df_trees_count.set_index("cb_num", drop=False)

	#Optional: Remove park districts (only available data from 2005 and 1995, and very small counts)
	#Or instead, do an inner merge
	df_trees_count = df_trees_count.dropna()
		
	# Merging the three datasets in one
	df_final = pd.concat([df_trees_count, df_air, health_data], axis=1, join="inner") #df_trees, 
	df_final['Geo Place Name'] = pd.Series(id_neighbourhood_dict)

	return df_final

def training_model_1(df_final):
	training_columns = [
						"count_2015",
						"area_km2",
						"counts_per_km2_2015",
						"Fine Particulate Matter (PM2.5)", 
						"Sulfur Dioxide (SO2)", 
						"Poverty",
						"Smoking",
						"Self_rep_health",
						"Unemployment",
						'park_percent'] 
	target_column = ["Avoidable_Asthma"]
	X = df_final[training_columns]
	y = df_final[target_column]
	X_train, X_test, y_train, y_test = train_test_split(X, y, 
											test_size=0.33, random_state=42)
	xgb_model_1 = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
	xgb_model_1.fit(X_train, y_train)
	y_pred = xgb_model_1.predict(X_test)
	mse_1=mean_squared_error(y_test, y_pred)
	#print("MSE: ",np.sqrt(mse_1))
	
	return xgb_model_1, y_test, y_pred, mse_1
	
def training_model_2(df_final):
	training_columns = [
						"count_2015",
						"area_km2",
						"counts_per_km2_2015",
						"Fine Particulate Matter (PM2.5)", 
						"Sulfur Dioxide (SO2)",
						"Poverty",
						"Smoking",
						"Self_rep_health",
						"Unemployment",
						'park_percent'] 
	target_column = ["Life_expectancy_rate"]
	X = df_final[training_columns]
	y = df_final[target_column]
	X_train, X_test, y_train, y_test = train_test_split(X, y, 
											test_size=0.33, random_state=42)
	xgb_model_2 = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
	xgb_model_2.fit(X_train, y_train)
	y_pred = xgb_model_2.predict(X_test)
	mse_2=mean_squared_error(y_test, y_pred)
	#print("MSE: ",np.sqrt(mse_2))
	
	return xgb_model_2, y_test, y_pred, mse_2
	
def plot_model_results(df_final, xgb_model_1, y_test_1, y_pred_1, mse_1, xgb_model_2, y_test_2, y_pred_2, mse_2):

	df_comparison_1 = y_test_1.copy()
	df_comparison_1['Predictions'] = y_pred_1
	df_comparison_1['ID'] = df_comparison_1.index
	df_comparison_1.columns = ['Real', 'Predicted', 'ID']
	df_comparison_1 = df_comparison_1.sort_index()

	df_comparison_2 = y_test_2.copy()
	df_comparison_2['Predictions'] = y_pred_2
	df_comparison_2['ID'] = df_comparison_2.index
	df_comparison_2.columns = ['Real', 'Predicted', 'ID']
	df_comparison_2 = df_comparison_2.sort_index()


	# Plot the two datasets together
	source1 = ColumnDataSource(data=df_comparison_1)
	ttps1=[("Real","@Real"),("Predicted","@Predicted")]
	fig1 = figure(height=300, width=600, tooltips=ttps1) #p is a standard way to call figures in Bokeh
	fig1.line("ID","Real",
				  source=source1,
				  legend_label="Real",
				  color='blue',
				  width=0.9)
	fig1.line("ID","Predicted",
				  source=source1,
				  legend_label="Predicted",
				  color='red',
				  width=0.9)
	#fig1.y_range.start = 0
	fig1.x_range.range_padding = 0.1
	fig1.xgrid.grid_line_color = None
	fig1.axis.minor_tick_line_color = None
	fig1.outline_line_color = None
	fig1.legend.location = "top_right"
	fig1.legend.orientation = "vertical"
	fig1.legend.title_text_font_style = "bold"
	fig1.legend.title_text_font_size = "20px"
	fig1.title.text="Results of Avoidable Asthma prediction"
	fig1.title.text_font_size = "15px"

	source2 = ColumnDataSource(data=df_comparison_2)
	ttps2=[("Real","@Real"),("Predicted","@Predicted")]
	fig2 = figure(height=300, width=600, tooltips=ttps2) #p is a standard way to call figures in Bokeh
	fig2.line("ID","Real",
				  source=source2,
				  legend_label="Real",
				  color='blue',
				  width=0.9)
	fig2.line("ID","Predicted",
				  source=source2,
				  legend_label="Predicted",
				  color='red',
				  width=0.9)
	#fig2.y_range.start = 0
	fig2.x_range.range_padding = 0.1
	fig2.xgrid.grid_line_color = None
	fig2.axis.minor_tick_line_color = None
	fig2.outline_line_color = None
	fig2.legend.location = "top_right"
	fig2.legend.orientation = "vertical"
	fig2.legend.title_text_font_style = "bold"
	fig2.legend.title_text_font_size = "20px"
	fig2.title.text="Results of Life Expectancy prediction"
	fig2.title.text_font_size = "15px"

	l1 = layout([[fig1]], sizing_mode='stretch_both')
	l2 = layout([[fig2]], sizing_mode='stretch_both')

	tab1 = Panel(child=l1,title="Avoidable Asthma Rate")
	tab2 = Panel(child=l2,title="Life Expectancy Rate")
	tabs = Tabs(tabs=[tab1, tab2])

	curdoc().add_root(tabs)
	
	return tabs
	
def model_interpretation(df_final, xgb_model_1, xgb_model_2):
	# Create a dataframe with the 1st model's parameters
	feature_df_1 = pd.DataFrame({'column':xgb_model_1.get_booster().feature_names, 'values':list(xgb_model_1.feature_importances_)})
	# Reorder it based on the values
	ordered_df_1 = feature_df_1.sort_values(by='values')
	my_range_1 = np.arange(1, 2*(len(feature_df_1.index))+1, 2)
	# Create a dataframe with the 2nd model's parameters
	feature_df_2 = pd.DataFrame({'column':xgb_model_2.get_booster().feature_names, 'values':list(xgb_model_2.feature_importances_)})
	# Reorder it based on the values
	ordered_df_2 = feature_df_2.sort_values(by='values')
	my_range_2 = np.arange(1.5, 2*(len(feature_df_2.index))+1.5, 2)
	
	fig = plt.figure(figsize=(3, 2))
	width = 0.5
	plt.hlines(y=my_range_1, xmin=0, xmax=ordered_df_1['values'], color='skyblue',linewidth=1)
	plt.hlines(y=my_range_2, xmin=0, xmax=ordered_df_2['values'], color='orange',linewidth=1)
	plt.plot(ordered_df_1['values'], my_range_1, "o", markersize=1, color='blue')
	plt.plot(ordered_df_2['values'], my_range_2, "o", markersize=1, color='red')
	plt.yticks([x+0.25 for x in my_range_1], ordered_df_1['column'])
	plt.title("Feature importances from the Models", loc='left', fontsize=5)
	plt.xlabel('Value of the parameter', fontsize=5)
	plt.ylabel('Parameters', fontsize=5)
	plt.xticks(fontsize=3)
	plt.yticks(fontsize=3)
	plt.legend(['Avoidable Asthma rate','Life Expectancy rate'], loc=4,prop={'size': 5})
	return fig
	
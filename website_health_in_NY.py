'''
Code for the project website using streamlit.
To view the website, run the code locally on your browser.
'''
import streamlit as st
import pandas as pd
import folium
import json
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import scipy
from matplotlib.pyplot import figure


from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import layout
from bokeh.models.widgets import Tabs, Panel
from bokeh.io import curdoc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

#import functions for the different datasets
#import modules.health_functions as health_func
#import modules.air_functions as air_func
#import modules.tree_functions as tree_func
#import modules.model_functions as ml_func
#import modules.corr_functions as cr_func

from PIL import Image


#
from matplotlib.backends.backend_agg import RendererAgg
matplotlib.use("agg")
_lock = RendererAgg.lock

#set to wide mode
st. set_page_config(layout="wide")
#remove warnings
st.set_option('deprecation.showPyplotGlobalUse', False)


#%% Here comes the code from "health_functions"

def health_data_on_map(health_data,geodata):
    data1=[]
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Poverty'],
                                        colorbar=dict(title='Poverty <br>(% below minimum)'),
                                        colorscale="sunset",
                                        hovertext=health_data["Name"],
                                        hoverinfo="text"
                                        ))
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Avoidable asthma'],
                                        colorbar=dict(title='Avoidable <br> Asthma <br> (nr. persons)'),
                                        colorscale="sunset",
                                        hovertext=health_data["Name"],
                                        hoverinfo="text"
                                        ))
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Self reported health'],
                                        colorbar=dict(title='Self reported <br> health (%)'),
                                        colorscale="sunset",
                                        hovertext=health_data["Name"],
                                        hoverinfo="text"
                                        ))
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Smoking'],
                                        colorbar=dict(title='Smoking (%)'),
                                        colorscale="sunset",
                                        hovertext=health_data["Name"],
                                        hoverinfo="text"
                                        ))
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Life expectancy'],
                                        colorbar=dict(title='Life expectancy <br>(age)'),
                                        colorscale="sunset",
                                        hovertext=health_data["Name"],
                                        hoverinfo="text"
                                        ))
    data1[0]['visible']=True
    data1[1]['visible']=False
    data1[2]['visible']=False
    data1[3]['visible']=False
    data1[4]['visible']=False

    layout=go.Layout(mapbox_style="carto-positron",
                    mapbox_zoom=9, mapbox_center = {"lat": 40.7, "lon": -73.86})

    #dropdown code from https://plot.ly/~empet/15237/choroplethmapbox-with-dropdown-menu/#/
    layout.update(updatemenus=list([
            dict(
                x=0.3,
                y=1,
                yanchor='top',
                buttons=list([
                    dict(
                        args=['visible', [True, False,False,False,False]],
                        label='Poverty',
                        method='restyle'
                    ),
                    dict(
                        args=['visible', [False, True, False,False,False]],
                        label='Avoidable Asthma',
                        method='restyle'
                    ),
                    dict(
                        args=['visible', [False,False, True,False,False]],
                        label='Self reported health',
                        method='restyle'),
                    dict(
                        args=['visible', [False,False, False,True,False]],
                        label='Smoking',
                        method='restyle'),
                    dict(
                        args=['visible', [False,False, False,False,True]],
                        label='Life expectancy',
                        method='restyle')]))]))
    fig = go.Figure(data=data1,layout=layout)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},title_text="Social factors and Health across NY District", title_x=0.5)
    return fig

# plotting the top and bottom 4 districts for poverty and relating the other features by ranking the 
# districts and showing the top 4 places in green and bottom 4 places in red
def poverty_ranking(health_data):
    #ranking the different features
    health_data['Poverty_rank']=health_data['Poverty'].rank(method='max')
    health_data["Unemployment_rank"]=health_data["Unemployment"].rank(method='max')
    health_data["Self_rep_health_rank"]=health_data["Self reported health"].rank(method='max')
    health_data["Smoking_rank"]=health_data["Smoking"].rank(method='max')
    health_data["Avoidable_Asthma_rank"]=health_data["Avoidable asthma"].rank(method='max')
    health_data["Life_expectancy_rate_rank"]=health_data["Life expectancy"].rank(method='max')
    
    
    health_data_top=health_data.nlargest(4, "Poverty_rank")
    health_data_min=health_data.nsmallest(4, "Poverty_rank")
    health_data_focus=pd.concat([health_data_top,health_data_min])
    fig, ax=plt.subplots(2,3,figsize=(15,10),sharey=True,sharex=True)
    ax=ax.flatten()
    
    colors = []
    for value in health_data_focus["Poverty_rank"]: # keys are the names of the boys
        if value>55:
            colors.append('g')
        elif value<=4:
            colors.append('r')
        else:
            colors.append('b')
    ax[0].barh(health_data_focus["Name"], health_data_focus["Poverty_rank"],color=colors)
    ax[0].title.set_text("Poverty")
    ax[0].xaxis.set_ticklabels([])
    
    colors = []
    for value in health_data_focus["Unemployment_rank"]: # keys are the names of the boys
        if value>55:
            colors.append('g')
        elif value<=4:
            colors.append('r')
        else:
            colors.append('b')
    ax[1].barh(health_data_focus["Name"], health_data_focus["Unemployment_rank"],color=colors)
    ax[1].title.set_text("Unemployment")
    ax[1].xaxis.set_ticklabels([])
    
    colors = []
    for value in health_data_focus["Self_rep_health_rank"]: # keys are the names of the boys
        if value>55:
            colors.append('g')
        elif value<=4:
            colors.append('r')
        else:
            colors.append('b')
    ax[2].barh(health_data_focus["Name"], health_data_focus["Self_rep_health_rank"],color=colors)
    ax[2].title.set_text("Self reported health")
    ax[2].xaxis.set_ticklabels([])
    
    colors = []
    for value in health_data_focus["Smoking_rank"]: # keys are the names of the boys
        if value>55:
            colors.append('g')
        elif value<=4:
            colors.append('r')
        else:
            colors.append('b')
    ax[3].barh(health_data_focus["Name"], health_data_focus["Smoking_rank"],color=colors)
    ax[3].title.set_text("Smoking")
    ax[3].xaxis.set_ticklabels([])
    
    colors = []
    for value in health_data_focus["Avoidable_Asthma_rank"]: # keys are the names of the boys
        if value>55:
            colors.append('g')
        elif value<=4:
            colors.append('r')
        else:
            colors.append('b')
    ax[4].barh(health_data_focus["Name"], health_data_focus["Avoidable_Asthma_rank"],color=colors)
    ax[4].title.set_text("Adult Asthma")
    ax[4].xaxis.set_ticklabels([])
    
    colors = []
    for value in health_data_focus["Life_expectancy_rate_rank"]: # keys are the names of the boys
        if value>55:
            colors.append('g')
        elif value<=4:
            colors.append('r')
        else:
            colors.append('b')
    ax[5].barh(health_data_focus["Name"], health_data_focus["Life_expectancy_rate_rank"],color=colors)
    ax[5].title.set_text("Life expectancy")
    ax[5].xaxis.set_ticklabels([])
    fig.suptitle("Ranking of socio-health factors in reference to Poverty")
    
    return fig


def regression_line(x,y):
    a=(sum(x[k]*y[k] for k in range(len(x)))-len(x)*np.mean(y)*np.mean(x))/(sum(x[k]**2 for k in range(len(x)))-len(x)*np.mean(x)**2)
    b=np.mean(y)-np.mean(x)*a
    y_hand=b+a*x
    return y_hand
def correlation_two_var(health_data):
    #normalize values to plot in one graph
    for i in health_data.columns[2:8]:
        name=i+"_norm"
        health_data[name] = [number/scipy.linalg.norm(health_data[i]) for number in health_data[i]]

    data=[]
    #avoidable asthma on x axis
    data.append(go.Scatter(name="Poverty",x=health_data["Avoidable asthma_norm"], y=health_data["Poverty_norm"],mode="markers",marker={'color':'blue'}))
    data.append(go.Scatter(name="Unemployment",x=health_data["Avoidable asthma_norm"], y=health_data["Unemployment_norm"],mode="markers",marker={'color':'red'}))
    data.append(go.Scatter(name="Self reported health",x=health_data["Avoidable asthma_norm"], y=health_data["Self reported health_norm"],mode="markers",marker={'color':'green'}))
    data.append(go.Scatter(name="Smoking",x=health_data["Avoidable asthma_norm"], y=health_data["Smoking_norm"],mode="markers",marker={'color':'orange'}))
    data.append(go.Scatter(name="Life expectancy",x=health_data["Avoidable asthma_norm"], y=health_data["Life expectancy_norm"],mode="markers",marker={'color':'purple'}))
    
    data.append(go.Scatter(x=np.array(health_data["Avoidable asthma_norm"]), y=regression_line(np.array(health_data["Avoidable asthma_norm"]),np.array(health_data["Poverty_norm"])),mode="lines",showlegend=False,line= dict(color="blue")))
    data.append(go.Scatter(x=np.array(health_data["Avoidable asthma_norm"]), y=regression_line(np.array(health_data["Avoidable asthma_norm"]),np.array(health_data["Unemployment_norm"])),mode="lines",showlegend=False,line= dict(color="red")))
    data.append(go.Scatter(x=np.array(health_data["Avoidable asthma_norm"]), y=regression_line(np.array(health_data["Avoidable asthma_norm"]),np.array(health_data["Self reported health_norm"])),mode="lines",showlegend=False,line= dict(color="green")))
    data.append(go.Scatter(x=np.array(health_data["Avoidable asthma_norm"]), y=regression_line(np.array(health_data["Avoidable asthma_norm"]),np.array(health_data["Smoking_norm"])),mode="lines",showlegend=False,line= dict(color="orange")))
    data.append(go.Scatter(x=np.array(health_data["Avoidable asthma_norm"]), y=regression_line(np.array(health_data["Avoidable asthma_norm"]),np.array(health_data["Life expectancy_norm"])),mode="lines",showlegend=False,line= dict(color="purple")))
    
    #life expectancy on x axis
    data.append(go.Scatter(name="Poverty",x=health_data["Life expectancy_norm"], y=health_data["Poverty_norm"],mode="markers",marker={'color':'blue'}))
    data.append(go.Scatter(name="Unemployment",x=health_data["Life expectancy_norm"], y=health_data["Unemployment_norm"],mode="markers",marker={'color':'red'}))
    data.append(go.Scatter(name="Self reported health",x=health_data["Life expectancy_norm"], y=health_data["Self reported health_norm"],mode="markers",marker={'color':'green'}))
    data.append(go.Scatter(name="Smoking",x=health_data["Life expectancy_norm"], y=health_data["Smoking_norm"],mode="markers",marker={'color':'orange'}))
    data.append(go.Scatter(name="Avoidable asthma",x=health_data["Life expectancy_norm"], y=health_data["Avoidable asthma_norm"],mode="markers",marker={'color':'purple'}))
    
    data.append(go.Scatter(x=np.array(health_data["Life expectancy_norm"]), y=regression_line(np.array(health_data["Life expectancy_norm"]),np.array(health_data["Poverty_norm"])),mode="lines",showlegend=False,line= dict(color="blue")))
    data.append(go.Scatter(x=np.array(health_data["Life expectancy_norm"]), y=regression_line(np.array(health_data["Life expectancy_norm"]),np.array(health_data["Unemployment_norm"])),mode="lines",showlegend=False,line= dict(color="red")))
    data.append(go.Scatter(x=np.array(health_data["Life expectancy_norm"]), y=regression_line(np.array(health_data["Life expectancy_norm"]),np.array(health_data["Self reported health_norm"])),mode="lines",showlegend=False,line= dict(color="green")))
    data.append(go.Scatter(x=np.array(health_data["Life expectancy_norm"]), y=regression_line(np.array(health_data["Life expectancy_norm"]),np.array(health_data["Smoking_norm"])),mode="lines",showlegend=False,line= dict(color="orange")))
    data.append(go.Scatter(x=np.array(health_data["Life expectancy_norm"]), y=regression_line(np.array(health_data["Life expectancy_norm"]),np.array(health_data["Avoidable asthma_norm"])),mode="lines",showlegend=False,line= dict(color="purple")))
    
    for i in range(10):
        data[i]['visible']=True
    for i in range(10,20):
        data[i]['visible']=False
    
    layout=go.Layout(
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(211,211,211,1)')
    
    layout.update(updatemenus=list([
            dict(
                x=1.2,
                y=0.3,
                yanchor='top',
                buttons=list([
                    dict(
                        args=['visible', [True,True,True,True,True,True,True,True,True,True,False,False,False,False,False,False,False,False,False,False]],
                        label='Avoidable Asthma',
                        method='restyle'
                    ),
                    dict(
                        args=['visible', [False,False,False,False,False,False,False,False,False,False, True,True,True,True,True,True,True,True,True,True]],
                        label='Life expectancy',
                        method='restyle')]))]))
    
    fig=go.Figure(data=data, layout=layout)
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, 
                      title_text="Correlation between Variables", 
                      title_x=0.4)
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    return fig

def pairplot_health(health_data):
    g=sns.pairplot(health_data.iloc[:,:8],kind="reg",plot_kws={'line_kws':{'color':'red'}})
    g.fig.suptitle("Pairplot of health and social variables", y=1.08)
    return g

def density_health(health_data):
    
    plt.figure(figsize=(13,5), dpi= 80)
    ax=sns.distplot((health_data["Poverty"]-np.mean(health_data["Poverty"]))/np.std(health_data["Poverty"]), color="dodgerblue", label="Poverty")
    ax=sns.distplot((health_data["Avoidable asthma"]-np.mean(health_data["Avoidable asthma"]))/np.std(health_data["Avoidable asthma"]),color="orange", label="Avoidable asthma")
    ax=sns.distplot((health_data["Self reported health"]-np.mean(health_data["Self reported health"]))/np.std(health_data["Self reported health"]), color="yellow", label="Self Reported Health")
    ax=sns.distplot((health_data["Unemployment"]-np.mean(health_data["Unemployment"]))/np.std(health_data["Unemployment"]), color="red", label="Unemployment")
    ax=sns.distplot((health_data["Smoking"]-np.mean(health_data["Smoking"]))/np.std(health_data["Smoking"]), color="green", label="Smoking")
    ax=sns.distplot((health_data["Life expectancy"]-np.mean(health_data["Life expectancy"]))/np.std(health_data["Life expectancy"]), color="purple", label="Life expectancy")
    
    plt.ylim(0, 1)
    
    # Decoration
    plt.title('Density Distribution of Health and Social features', fontsize=22)
    ax.set(xlabel='Normalized distribution', ylabel='Density')
    plt.legend()
    fig=ax.get_figure()
    return fig


#%%% Here comes the code from air_functions

R = 6371 #Earth radius, km for grid mapping


def plot_air_equal(df):
    ax=df.groupby(['Geo Place Name']).size().sort_values(ascending=False).plot(kind='barh',figsize=(3,6))
    plt.title("District data distribution")
    plt.yticks(size=3)
    fig=ax.get_figure()
    return fig

def get_neighbourhood_name(id_num,df):
    return df[df['Geo Join ID'] == id_num]['Geo Place Name'].iloc[0]

def fine_particle_matter_on_map(df,geo):
    # Modifying the coordinates by subtstituting the id with the name of the District to show in the map. 
    # get_neighbourhood_name is a function implemented at the beginning of this section.
    for feature in geo['features']:
        if int(feature['id']) > 0:
            feature['id'] = get_neighbourhood_name(int(feature['id']),df)
    Name = 'Fine Particulate Matter (PM2.5)'
    df_temp = df[df["Name"] == Name] # select specific Name
    measure_info = df_temp['Measure Info'].iloc[0]
    df_temp = df_temp.groupby(['Start Year','Geo Place Name']).mean() # calculate average per year
    df_temp = df_temp.reset_index(level=[0,1]) # reset indexes
    locations = list(set(df_temp['Geo Place Name'])) # save location names
    
    data=[]
    button_settings_list = []
    iterator = 0
    for year in sorted(set(df_temp['Start Year']), reverse=True):
        df_sub_temp = df_temp[df_temp['Start Year']==year]
        data.append(go.Choroplethmapbox(geojson=geo, 
                                        locations=locations, 
                                        z=df_sub_temp['Data Value'],
                                        colorbar=dict(title=dict(text=Name+" - "+measure_info, side="right")),
                                        colorscale="Reds",
										zmin=5,
										zmax=19

                                        ))
        setting_dict = {}
        visibility_list = [False]*(len(set(df_temp['Start Year'])))
        visibility_list[iterator] = True
        iterator += 1
        setting_dict['args']=['visible', visibility_list]
        setting_dict['label']=str(year)
        setting_dict['method']='restyle'
        button_settings_list.append(setting_dict)
        
    layout=go.Layout(mapbox_style="carto-positron",
                      mapbox_zoom=9.2, mapbox_center = {"lat": 40.7, "lon": -73.86})
    layout.update(updatemenus=list([
            dict(
                x=-0.05,
                y=1,
                yanchor='top',
                buttons=list(button_settings_list))]))
    fig = go.Figure(data=data,layout=layout)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    return fig

def sulfur_on_map(df,geo):
    for feature in geo['features']:
        if int(feature['id']) > 0:
            feature['id'] = get_neighbourhood_name(int(feature['id']),df)
    Name = 'Sulfur Dioxide (SO2)'
    df_temp = df[df["Name"] == Name] # select specific Name
    measure_info = df_temp['Measure Info'].iloc[0]
    df_temp = df_temp.groupby(['Start Year','Geo Place Name']).mean() # calculate average per year
    df_temp = df_temp.reset_index(level=[0,1]) # reset indexes
    locations = list(set(df_temp['Geo Place Name'])) # save location names
    
    data=[]
    button_settings_list = []
    iterator = 0
    for year in sorted(set(df_temp['Start Year']), reverse=True):
        df_sub_temp = df_temp[df_temp['Start Year']==year]
        data.append(go.Choroplethmapbox(geojson=geo, 
                                        locations=locations, 
                                        z=df_sub_temp['Data Value'],
                                        colorbar=dict(title=dict(text=Name+" - "+measure_info, side="right")),
                                        colorscale="YlOrBr",
										zmin=0,
										zmax=12

                                        ))
        setting_dict = {}
        visibility_list = [False]*(len(set(df_temp['Start Year'])))
        visibility_list[iterator] = True
        iterator += 1
        setting_dict['args']=['visible', visibility_list]
        setting_dict['label']=str(year)
        setting_dict['method']='restyle'
        button_settings_list.append(setting_dict)
        
    layout=go.Layout(mapbox_style="carto-positron",
                      mapbox_zoom=9.2, mapbox_center = {"lat": 40.7, "lon": -73.86})
    layout.update(updatemenus=list([
            dict(
                x=-0.05,
                y=1,
                yanchor='top',
                buttons=list(button_settings_list))]))
    fig = go.Figure(data=data,layout=layout)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    
    return fig

def fine_particle_over_time(df):
    # Visualization of all districts for single variable

    fig_dict = {}
    layout_dict = {}
    tab_dict = {}
    
    Name = 'Fine Particulate Matter (PM2.5)'
    for Geo_Place_Name in set(df['Geo Place Name']):
        fig_dict[Geo_Place_Name] = figure(height=500, width=600,x_axis_type='datetime')
        asthma = df[df['Name'] == Name].sort_values('Date', axis=0)
        asthma = asthma[asthma['Geo Place Name'] == Geo_Place_Name]
        asthma = asthma[asthma['Name'] == Name]
        source = ColumnDataSource(data=asthma)
        fig_dict[Geo_Place_Name].line(x='Start_Date', 
                                      y='Data Value',
                                      source=source,
                                      width=0.9)
        fig_dict[Geo_Place_Name].circle(x='Start_Date', 
                                        y='Data Value',
                                        source=source)
        fig_dict[Geo_Place_Name].add_tools(HoverTool(tooltips=[("Data Value", '@{Data Value}'), ("Date", "@Start_Date{%F}")],
                                        formatters={'@Start_Date':'datetime'}))
        fig_dict[Geo_Place_Name].x_range.range_padding = 0.1
        fig_dict[Geo_Place_Name].xgrid.grid_line_color = None
        fig_dict[Geo_Place_Name].axis.minor_tick_line_color = None
        fig_dict[Geo_Place_Name].outline_line_color = None
    
        title = Name+" - "+Geo_Place_Name
        fig_dict[Geo_Place_Name].title.text = title
        fig_dict[Geo_Place_Name].title.text_font_size = "15px"
        fig_dict[Geo_Place_Name].xaxis.axis_label = 'Date'
        fig_dict[Geo_Place_Name].yaxis.axis_label = df[df['Name']==Name]['Measure Info'].iloc[0]
        
        layout_dict[Geo_Place_Name] = layout([[fig_dict[Geo_Place_Name]]], sizing_mode='stretch_both')
    
        tab_dict[Geo_Place_Name] = Panel(child=layout_dict[Geo_Place_Name], title=Geo_Place_Name)
        
    tabs = Tabs(tabs=list(tab_dict.values()))
    curdoc().add_root(tabs)
    
    return tabs

def sulfur_over_time(df):
    # Visualization of all districts for single variable

    fig_dict = {}
    layout_dict = {}
    tab_dict = {}
    
    Name = 'Sulfur Dioxide (SO2)'
    for Geo_Place_Name in set(df['Geo Place Name']):
        fig_dict[Geo_Place_Name] = figure(height=500, width=600,x_axis_type='datetime')
        asthma = df[df['Name'] == Name].sort_values('Date', axis=0)
        asthma = asthma[asthma['Geo Place Name'] == Geo_Place_Name]
        asthma = asthma[asthma['Name'] == Name]
        source = ColumnDataSource(data=asthma)
        fig_dict[Geo_Place_Name].line(x='Start_Date', 
                                      y='Data Value',
                                      source=source,
                                      width=0.9)
        fig_dict[Geo_Place_Name].circle(x='Start_Date', 
                                        y='Data Value',
                                        source=source)
        fig_dict[Geo_Place_Name].add_tools(HoverTool(tooltips=[("Data Value", '@{Data Value}'), ("Date", "@Start_Date{%F}")],
                                        formatters={'@Start_Date':'datetime'}))
        fig_dict[Geo_Place_Name].x_range.range_padding = 0.1
        fig_dict[Geo_Place_Name].xgrid.grid_line_color = None
        fig_dict[Geo_Place_Name].axis.minor_tick_line_color = None
        fig_dict[Geo_Place_Name].outline_line_color = None
    
        title = Name+" - "+Geo_Place_Name
        fig_dict[Geo_Place_Name].title.text = title
        fig_dict[Geo_Place_Name].title.text_font_size = "15px"
        fig_dict[Geo_Place_Name].xaxis.axis_label = 'Date'
        fig_dict[Geo_Place_Name].yaxis.axis_label = df[df['Name']==Name]['Measure Info'].iloc[0]
        
        layout_dict[Geo_Place_Name] = layout([[fig_dict[Geo_Place_Name]]], sizing_mode='stretch_both')
    
        tab_dict[Geo_Place_Name] = Panel(child=layout_dict[Geo_Place_Name], title=Geo_Place_Name)
        
    tabs = Tabs(tabs=list(tab_dict.values()))
    curdoc().add_root(tabs)
    
    return tabs



#%% Here comes the code from "corr_functions"

def merge_data(df_air_,df_health_,df_trees_):
    df_air=df_air_.copy()
    df_health=df_health_.copy()
    df_trees=df_trees_.copy()
    # Helper fuctions and dict for air dataset
    def get_neighbourhood_name(id_num, df):
        return df[df['Geo Join ID'] == id_num]['Geo Place Name'].iloc[0]
    
    def get_neighbourhoo_id(neighbourhood_name, df):
        return df[df['Geo Place Name'] == id_num]['Geo Join ID'].iloc[0]
    
    # Usage: id_neighbourhood_dict[**insert Districtd ID here**] -> get District Name
    id_neighbourhood_dict = {}
    for id_num in set(df_air['Geo Join ID']):
        id_neighbourhood_dict[id_num] = df_air[df_air['Geo Join ID'] == id_num]['Geo Place Name'].iloc[0]
        
    df_air['Year'] = df_air['Start_Date'].dt.to_period('Y').astype('str')
    df_air = df_air[df_air['Year']=='2015']
    df_air = df_air[['Name', 'Measure', 'Measure Info', 'Geo Join ID', 'Geo Place Name', 'Data Value']]
        
    df_air_col_1 = df_air[df_air['Name']=='Fine Particulate Matter (PM2.5)']
    df_air_col_2 = df_air[df_air['Name']=='Sulfur Dioxide (SO2)']
    df_air_col_1 = df_air_col_1.groupby('Geo Join ID').mean()
    df_air_col_1.columns = ['Fine Particulate Matter (PM2.5)'] 
    df_air_col_2 = df_air_col_2.groupby('Geo Join ID').mean()
    df_air_col_2.columns = ['Sulfur Dioxide (SO2)']


    df_health=df_health.set_index('ID')
    df_health.index = df_health.index.astype('int64')

    #Index by the district for the tree counts and areas
    df_trees = df_trees.set_index("cb_num", drop=False)

    
    # Merging the four datasets in one
    df_air_trees_health = pd.concat([df_trees, df_air_col_1, df_air_col_2,df_health[["Avoidable asthma","Poverty","Unemployment","Self reported health","Smoking","Life expectancy"]]], axis=1, join="inner")
    df_air_trees_health['Geo Place Name'] = pd.Series(id_neighbourhood_dict)
    #define the columns that we want to see the density from
    #df_dens_plot=df_air_trees_health.copy()
    #df_dens_plot=df_dens_plot.drop(['count_2005','count_1995','counts_per_km2_2005','counts_per_km2_1995','Geo Place Name'],axis=1)
    #df_dens_plot.rename(columns = {'count_2015':'Tree count 2015', 'counts_per_km2_2015':'Tree density 2015 (counts pr km2)','parkarea [km^2]':'Parkarea [km2]','area_km2':'Area [km2]','park_percent':'Park percentage', 'Avoidable_Asthma':'Avoidable asthma','Life_expectancy_rate':'Life expectancy','Self_rep_health':'Self reported health'}, inplace = True)

    
    return df_air_trees_health

def air_health_tree_correlation(df_air,df_health,df_trees):
    
    df_dens_plot=merge_data(df_air,df_health,df_trees)
    df_dens_plot=df_dens_plot.drop(['count_2005','count_1995','counts_per_km2_2005','counts_per_km2_1995','Geo Place Name'],axis=1)
    df_dens_plot.rename(columns = {'count_2015':'Tree count 2015', 'counts_per_km2_2015':'Tree density 2015 (counts pr km2)','parkarea [km^2]':'Parkarea [km2]','area_km2':'Area [km2]','park_percent':'Park percentage'}, inplace = True)

    
    #correlation plot
    corr = df_dens_plot.iloc[:,1:].corr()
    plt.figure(figsize = (15,8))
    ax = sns.heatmap(corr,vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=256), square=True,annot=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,horizontalalignment='right')
    ax.set_title('Correlation matrix of Air vs Tree vs Health dataframe')
    
    fig=ax.get_figure()
    return fig

def r_square(df_air,df_health,df_trees):
    df_dens_plot=merge_data(df_air,df_health,df_trees)
    df_dens_plot=df_dens_plot.drop(['count_2005','count_1995','counts_per_km2_2005','counts_per_km2_1995','Geo Place Name'],axis=1)
    df_dens_plot.rename(columns = {'count_2015':'Tree count 2015', 'counts_per_km2_2015':'Tree density 2015 (counts pr km2)','parkarea [km^2]':'Parkarea [km2]','area_km2':'Area [km2]','park_percent':'Park percentage'}, inplace = True)

    
    #correlation plot
    corr = df_dens_plot.iloc[:,1:].corr()
    
    r_square=corr**2
    plt.figure(figsize = (15,8))
    ax = sns.heatmap(r_square,vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=256), square=True,annot=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,horizontalalignment='right')
    ax.set_title('Correlation matrix of Air vs Tree vs Health density dataframe')
    
    fig=ax.get_figure()
    return fig

#This code comes from: https://github.com/dylan-profiler/heatmaps/blob/master/heatmap/heatmap.py
#by Drazen Zaric
def heatmap(x, y, **kwargs):
    #caching.clear_cache()
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = [1]*len(x)

    if 'palette' in kwargs:
        palette = kwargs['palette']
        n_colors = len(palette)
    else:
        n_colors = 256 # Use 256 colors for the diverging color palette
        palette = sns.color_palette("Blues", n_colors) 

    if 'color_range' in kwargs:
        color_min, color_max = kwargs['color_range']
    else:
        color_min, color_max = min(color), max(color) # Range of values that will be mapped to the palette, i.e. min and max possible correlation

    def value_to_color(val):
        if color_min == color_max:
            return palette[-1]
        else:
            val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            ind = int(val_position * (n_colors - 1)) # target index in the color palette
            return palette[ind]

    if 'size' in kwargs:
        size = kwargs['size']
    else:
        size = [1]*len(x)

    if 'size_range' in kwargs:
        size_min, size_max = kwargs['size_range'][0], kwargs['size_range'][1]
    else:
        size_min, size_max = min(size), max(size)

    size_scale = kwargs.get('size_scale', 500)

    def value_to_size(val):
        if size_min == size_max:
            return 1 * size_scale
        else:
            val_position = (val - size_min) * 0.99 / (size_max - size_min) + 0.01 # position of value in the input range, relative to the length of the input range
            val_position = min(max(val_position, 0), 1) # bound the position betwen 0 and 1
            return val_position * size_scale
    if 'x_order' in kwargs: 
        x_names = [t for t in kwargs['x_order']]
    else:
        x_names = [t for t in sorted(set([v for v in x]))]
    x_to_num = {p[1]:p[0] for p in enumerate(x_names)}

    if 'y_order' in kwargs: 
        y_names = [t for t in kwargs['y_order']]
    else:
        y_names = [t for t in sorted(set([v for v in y]))]
    y_to_num = {p[1]:p[0] for p in enumerate(y_names)}

    
    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x10 grid
    plt.figure(figsize=(8,6),dpi=80)
    ax = plt.subplot(plot_grid[:,:-1]) # Use the left 14/15ths of the grid for the main plot

    marker = kwargs.get('marker', 's')

    kwargs_pass_on = {k:v for k,v in kwargs.items() if k not in [
         'color', 'palette', 'color_range', 'size', 'size_range', 'size_scale', 'marker', 'x_order', 'y_order', 'xlabel', 'ylabel'
    ]}

    ax.scatter(
        x=[x_to_num[v] for v in x],
        y=[y_to_num[v] for v in y],
        marker=marker,
        s=[value_to_size(v) for v in size], 
        c=[value_to_color(v) for v in color],
        **kwargs_pass_on
    )
    ax.set_xticks([v for k,v in x_to_num.items()])
    ax.set_xticklabels([k for k in x_to_num], rotation=45, horizontalalignment='right')
    ax.set_yticks([v for k,v in y_to_num.items()])
    ax.set_yticklabels([k for k in y_to_num])

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)

    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5])
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])
    ax.set_facecolor('#F1F1F1')

    ax.set_xlabel(kwargs.get('xlabel', ''))
    ax.set_ylabel(kwargs.get('ylabel', ''))
    

    # Add color legend on the right side of the plot
    if color_min < color_max:
        ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot

        col_x = [0]*len(palette) # Fixed x coordinate for the bars
        bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

        bar_height = bar_y[1] - bar_y[0]
        ax.barh(
            y=bar_y,
            width=[5]*len(palette), # Make bars 5 units wide
            left=col_x, # Make bars start at 0
            height=bar_height,
            color=palette,
            linewidth=0
        )
        ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
        ax.grid(False) # Hide grid
        ax.set_facecolor('white') # Make background white
        ax.set_xticks([]) # Remove horizontal ticks
        ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
        ax.yaxis.tick_right() # Show vertical ticks on the right 
        fig=ax.get_figure()
        return fig
    else:
        fig=ax.get_figure()
        return fig


def corrplot(df_air,df_health,df_trees, size_scale=500, marker='s'):
    df_dens_plot=merge_data(df_air,df_health,df_trees)
    df_dens_plot=df_dens_plot.drop(['count_2005','count_1995','counts_per_km2_2005','counts_per_km2_1995','Geo Place Name'],axis=1)
    df_dens_plot.rename(columns = {'count_2015':'Tree count 2015', 'counts_per_km2_2015':'Tree density 2015 (counts pr km2)','parkarea [km^2]':'Parkarea [km2]','area_km2':'Area [km2]','park_percent':'Park percentage'}, inplace = True)

    data=df_dens_plot.iloc[:,1:].corr()
    corr = pd.melt(data.reset_index(), id_vars='index').replace(np.nan, 0)
    corr.columns = ['x', 'y', 'value']
    return heatmap(
        corr['x'], corr['y'],
        color=corr['value'], color_range=[-1, 1],
        palette=sns.diverging_palette(20, 220, n=256),
        size=corr['value'].abs(), size_range=[0,1],
        marker=marker,
        x_order=data.columns,
        y_order=data.columns[::-1],
        size_scale=size_scale
    )

def density_all(df_air,df_health,df_trees):
    df_dens_plot=merge_data(df_air,df_health,df_trees)
    df_dens_plot=df_dens_plot.drop(['count_2005','count_1995','counts_per_km2_2005','counts_per_km2_1995','Geo Place Name'],axis=1)
    df_dens_plot.rename(columns = {'count_2015':'Tree count 2015', 'counts_per_km2_2015':'Tree density 2015 (counts pr km2)','parkarea [km^2]':'Parkarea [km2]','area_km2':'Area [km2]','park_percent':'Park percentage'}, inplace = True)

    #make the plot
    def make_plot(title, hist, edges, x, pdf):
        p = figure(title=title, tools='', background_fill_color="#fafafa")
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
               fill_color="navy", line_color="white", alpha=0.5)
        p.line(x, pdf, line_color="#ff8888", line_width=4, alpha=0.7, legend_label="Density")
    
        p.y_range.start = 0
        p.legend.location = "center_right"
        p.legend.background_fill_color = "#fefefe"
        p.xaxis.axis_label = 'x'
        p.yaxis.axis_label = 'Pr(x)'
        p.grid.grid_line_color="white"
        return p
    
    fig_dict = {}
    layout_dict = {}
    tab_dict = {}
    
    #rename columns
    
    for Name in df_dens_plot.columns[1:]:
        # Normal Distribution
        mu=np.mean(df_dens_plot[Name])
        sigma=np.std(df_dens_plot[Name])
        hist, edges = np.histogram(df_dens_plot[Name], density=True, bins=10)
        x = np.linspace(min(df_dens_plot[Name]), max(df_dens_plot[Name]), 1000)
    
        density = scipy.stats.gaussian_kde(df_dens_plot[Name])
        pdf=density(x)
    
        p1 = make_plot(f"Normal Distribution (μ={mu}, σ={sigma})", hist, edges, x, pdf)
        title = "Density distribution of "+Name
        p1.title.text = title
        p1.title.text_font_size = "15px"
        p1.xaxis.axis_label = 'Distribution'
        p1.yaxis.axis_label = 'Density'
        fig_dict[Name]=p1
    
        layout_dict[Name] = layout([fig_dict[Name]], sizing_mode='stretch_both')
    
        tab_dict[Name] = Panel(child=layout_dict[Name], title=Name)
    
    
    tabs = Tabs(tabs=list(tab_dict.values()))
    curdoc().add_root(tabs)
    
    return tabs

def pairplots(df_air,df_health,df_trees):
    df_dens_plot=merge_data(df_air,df_health,df_trees)
    df_dens_plot=df_dens_plot.drop(['count_2005','count_1995','counts_per_km2_2005','counts_per_km2_1995','Geo Place Name'],axis=1)
    df_dens_plot.rename(columns = {'count_2015':'Tree count 2015', 'counts_per_km2_2015':'Tree density 2015 (counts pr km2)','parkarea [km^2]':'Parkarea [km2]','area_km2':'Area [km2]','park_percent':'Park percentage'}, inplace = True)

    plt.figure(figsize = (15,8))
    g = sns.pairplot(df_dens_plot, kind="reg",plot_kws={'line_kws':{'color':'red'}})
    g.fig.suptitle("Pairplot for Air, Tree and Health data", y=1.02)

    return g


#%% Here comes the code from tree_functions

def to_cartesian(latlon):
    """
    Convert lat-lon data into cartesian
    """
    R = 6371
    lat,lon = np.radians(latlon.T)
    clat = np.cos(lat)
    return R*np.column_stack([clat*np.cos(lon), clat*np.sin(lon), np.sin(lat)])

def density_estimate(coords, k, extent, grid_res=300, logscale=True, q=0.99, year=None):
    from matplotlib.colors import LogNorm
    from scipy.spatial import cKDTree as KDTree
    tree = KDTree(to_cartesian(coords))
    x0,x1,y0,y1 = extent
    dx = (x1-x0)/grid_res
    dy = (y1-y0)/grid_res
    g = np.moveaxis(np.mgrid[x0:x1:dx,y0:y1:dy], 0, -1)
    r = tree.query(to_cartesian(g.reshape(-1,2)), k, workers=-1)[0][:,-1]
    d = (k/np.pi)*r**-2
    d = d.reshape(*g.shape[:-1])
    
    if logscale:
        norm = LogNorm()
    else:
        norm = None
    fig=plt.figure(figsize=[20,14])
    plt.imshow(d, origin='lower', extent=(y0,y1,x0,x1), vmax=np.quantile(d, q), aspect='auto', norm=norm, cmap=plt.cm.inferno)
    plt.colorbar()
    if year == None:
        plt.title("Density estimate of trees recorded in NYC tree census, in trees / km$^2$")
    else:
        plt.title(f"{year}: Density estimate of trees recorded in NYC tree census, in trees / km$^2$")
    plt.grid(False)
    return fig

def tree_density(df):
    density_estimate(df[['latitude', 'longitude']].dropna().values, 25, [40.48,40.92,-74.3,-73.65], grid_res=800, logscale=False, year=2015)

def tree_count(df,geo):
    #Plot maps using plotly
    df_counts=df.copy()
    data=[]
    data.append(go.Choroplethmapbox(geojson=geo, 
                                        locations=df_counts.cb_num, 
                                        z=df_counts.counts_per_km2_1995,
                                        colorbar=dict(title='1995'),
                                        colorscale="YlGn",
                                        zmin=0,
                                        zmax=170
                                        ))
    data.append(go.Choroplethmapbox(geojson=geo, 
                                        locations=df_counts.cb_num,
                                        z=df_counts.counts_per_km2_2005,
                                        colorbar=dict(title='2005'),
                                        colorscale="YlGn",
                                        zmin=0,
                                        zmax=170
                                        ))
    data.append(go.Choroplethmapbox(geojson=geo, 
                                        # locations=df_trees_2015_cln.cb_num.value_counts().index, 
                                        # z=df_trees_2015_cln.nta.value_counts().values,
                                        locations=df_counts.cb_num,
                                        z=df_counts.counts_per_km2_2015,
                                        colorbar=dict(title='2015'),
                                        colorscale="YlGn",
                                        zmin=0,
                                        zmax=170
                                        ))
    data[0]['visible']=True
    data[1]['visible']=False
    data[2]['visible']=False
    layout=go.Layout(mapbox_style="carto-positron",
                      mapbox_zoom=9, mapbox_center = {"lat": 40.7, "lon": -73.86})
    #dropdown code from https://plot.ly/~empet/15237/choroplethmapbox-with-dropdown-menu/#/
    layout.update(updatemenus=list([
            dict(
                x=0.15,
                y=1,
                yanchor='top',
                buttons=list([
                    dict(
                        args=['visible', [True, False,False]],
                        label='Year: 1995',
                        method='restyle'
                    ),
                    dict(
                        args=['visible', [False, True, False]],
                        label='Year: 2005',
                        method='restyle'
                    ),
                    dict(
                        args=['visible', [False,False, True]],
                        label='Year: 2015',
                        method='restyle')]))]))
    fig = go.Figure(data=data,layout=layout)
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, title_text="Tree Density by NY District", title_x=0.5)
    return fig

def tree_change(df,geo):
    df_trees_2005_cln=df[1]
    df_trees_2015_cln=df[2]
    
    nta_count_2005=pd.DataFrame(df_trees_2005_cln.cb_num.value_counts()).reset_index()
    nta_count_2015=pd.DataFrame(df_trees_2015_cln.cb_num.value_counts()).reset_index()
    nta_count_diff=nta_count_2015.merge(nta_count_2005, on='index')
    nta_count_diff['change']=nta_count_diff['cb_num_x']-nta_count_diff['cb_num_y']
    
    data=[]
    data.append(go.Choroplethmapbox(geojson=geo ,
                                        locations=df_trees_2015_cln.cb_num.value_counts().index, 
                                        z=nta_count_diff.change.values,
                                        colorbar=dict(title='2015-2005 Change'),
                                        colorscale=[[0, "red"],
                                                    [0.15,'yellow'],
                                                   [1, "green"]]
                                        ))
    
    layout=go.Layout(mapbox_style="carto-positron",
                      mapbox_zoom=9, mapbox_center = {"lat": 40.7, "lon": -73.86})
    
    
    fig = go.Figure(data=data,layout=layout)
    fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0}, title_text="Change in tree count from 2005 to 2015", title_x=0.5)
    return fig

def tree_change_per_district(df):
    df_counts=df
    df_counts_dis = df_counts.copy()
    df_counts_dis["2015_2005_diff"] = df_counts_dis.counts_per_km2_2015 - df_counts_dis.counts_per_km2_2005
    df_counts_dis = df_counts_dis.sort_values(by="2015_2005_diff", ascending=False)
    df_counts_dis = df_counts_dis.dropna()
    colors = ["red" if x < 0 else "green" for x in df_counts_dis["2015_2005_diff"]]
    df_counts_dis["2015_2005_diff"] =np.round(df_counts_dis["2015_2005_diff"], 1)
    ax = df_counts_dis.plot.barh(x="cb_num", y="2015_2005_diff", figsize=(10,15), color=colors)
    ax.bar_label(ax.containers[0])
    
    ax.set_title("Change in the tree density from 2005 to 2015 by district")
    ax.set_ylabel("District number")
    ax.set_xlabel(f"Change in tree density $(trees/km^2)$")
    plt.tight_layout()
    fig=ax.get_figure()
    return fig







#load the data
geodata =  json.load(open('./data/geo_files/json/CD.json', 'r'))
geo = json.load(open("./data/geo_files/json/community_districts_updated.geojson"))

@st.cache(allow_output_mutation=True)
def load_health_data():
    health_data = pd.read_csv('./data/health/2015_Health2.csv',sep=";")
    # filtering the data according to required subsets
    health_data.drop(labels=[0,1,2,3,4,5],axis=0,inplace=True)
    health_data=health_data.reset_index()
    focus_health=["ID","Name","Avoidable_Asthma","Poverty","Unemployment","Self_rep_health","Smoking","Life_expectancy_rate"]
    health_data=health_data.loc[:, health_data.columns.isin(focus_health)]
    health_data=health_data.iloc[:59]
    return health_data

@st.cache
def load_tree_data():
    df_trees_1995 = pd.read_csv("./data/trees/trim_final_clean_new_york_tree_census_1995.csv")
    df_trees_2005 = pd.read_csv("./data/trees/trim_final_clean_new_york_tree_census_2005.csv")
    df_trees_2015 = pd.read_csv("./data/trees/trim_final_clean_new_york_tree_census_2015.csv")
    
    return df_trees_1995,df_trees_2005,df_trees_2015

def tree_density_data(geo,df):
    df_counts = pd.read_csv("./data/trees/areas_tree_counts.csv")
    return df_counts

@st.cache(allow_output_mutation=True)
def load_airquality_data():
    df = pd.read_csv("./data/air_quality/Air_Quality.csv")
    df['Start_Date'] =  pd.to_datetime(df['Start_Date'], format='%m/%d/%Y')
    df['Start Year'] = pd.DatetimeIndex(df['Start_Date']).year
    df['Date'] = pd.to_datetime(df['Start_Date'], format='%M/%d/%Y')
    return df


#%% Here comes the code from "model_functions

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
						"Self reported health",
						"Unemployment",
						'park_percent'] 
	target_column = ["Avoidable asthma"]
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
						"Self reported health",
						"Unemployment",
						'park_percent'] 
	target_column = ["Life expectancy"]
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
	tab2 = Panel(child=l2,title="Life Expectancy")
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
	plt.legend(['Avoidable Asthma rate','Life Expectancy'], loc=4,prop={'size': 5})
	return fig
	

#%%%%%% Map Plot

#Map of NY
def ny_map():
    map_hooray = folium.Map(location=[40.7, -73.86],
                        width="%60",
                        height="%70",
                        zoom_start = 10) # Uses lat then lon. The bigger the zoom number, the closer in you get
    return map_hooray # Calls the map to display

def ny_map_trees(df_trees_2015):
    tree_type_list=df_trees_2015["spc_latin"].unique()
    
    #get ten most common tree-types:
    n = 10
    tree_type_list=df_trees_2015["spc_latin"].value_counts()[:n].index.tolist()
    
    # Filter the DF for rows, then columns, then remove NaNs
    heat_df = df_trees_2015[df_trees_2015["spc_latin"].isin(tree_type_list)] # Reducing data size so it runs faster
    
    
    heat_df = heat_df[['latitude', 'longitude']]
    heat_df = heat_df.dropna(axis=0, subset=['latitude','longitude'])
    
    
    m = folium.Map(
        location=[40.706005, -74.008827],
        width="%60",
        height="%70",
        zoom_start=15,
    )
    
    # List comprehension to make out list of lists
    heat_data = [[row['latitude'],row['longitude']] for index, row in heat_df.iterrows()]
    
    # Plot it on the map
    HeatMap(heat_data,radius = 7, min_opacity = 3,gradient={0.3:'darkgreen',.6: 'green',0.75:'yellow', .9: 'orange'}).add_to(m)
    
    #def plotDot(point):
    #    '''input: series that contains a numeric named latitude and a numeric named longitude
    #    this function creates a CircleMarker and adds it to your this_map'''
    #    folium.CircleMarker(location=[point.latitude, point.longitude],
    #                        radius=2,
    #                        weight=5).add_to(m)
    
    #use df.apply(,axis=1) to "iterate" through every row in your dataframe
    #heat_df.apply(plotDot, axis = 1)
    
    # Display the map
    return m

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Data load
# Load the data
health_data = load_health_data()
health_data.rename(columns = {'Avoidable_Asthma':'Avoidable asthma', 'Self_rep_health':'Self reported health','Life_expectancy_rate':'Life expectancy'}, inplace = True)

tree_data=load_tree_data()
tree_dens_data=tree_density_data(geo,tree_data)
airquality_data=load_airquality_data()

#load images and assets
correlation_im = Image.open("./data/images/corr.png")


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%CSS LAYOUT 

#def local_css(file_name):
#    with open(file_name) as f:
#        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

#local_css("modules/style.css")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%PAGE SETUP

#set title
st.title('Do trees influence the health of NY citizens?')

# Add the overview of the article to the sidebar:
st.sidebar.markdown("# Sections", unsafe_allow_html=True)
st.sidebar.markdown("* [Introduction](#introduction)", unsafe_allow_html=True)
st.sidebar.markdown("* [Data Exploration](#data-exploration)", unsafe_allow_html=True)
st.sidebar.markdown("* [Correlation Analysis between airquality, health and tree density](#correlation-analysis-between-airquality-health-and-tree-density)", unsafe_allow_html=True)
st.sidebar.markdown("* [Machine Learning model](#machine-learning-model)", unsafe_allow_html=True)
st.sidebar.markdown("* [Conclusion](#conclusion)", unsafe_allow_html=True)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
st.header("Introduction")
left_column1, right_column1 = st.columns((2,1))
with left_column1:
    st.markdown("Imagine yourself, sitting on a bench and listening to the wind, \
                that softly strokes and shuffles the soft, green leaves of the tree \
                standing next to the bench. You look up and are blinded by a few \
                beams of afternoon sun cheekily peeking through the umbrella \
                of leaves, forcing you to close your eyes and give in to \
                the moment. There is the song of a blackbird, singing \
                to the scene from one of the branches of the tree \
                and the distinct reply of another blackbird \
                further away. You also hear the buzzing of insects that \
                feed on the trees’ blossoms. Then, suddenly, the loud roar \
                of a combustive engine –  you open your eyes and are back \
                in the city, sitting on one of the few benches next to the main road.")
                    
    st.markdown("We all know the calming effect that nature can have on us and \
                it is proven, that street trees are no exception for that. Scientist from the UK\
                that studied the effect of street trees on mental health  \
                in 2020 found out, that the presence of street trees can even \
                be more efficient than antidepressiva ([Marselle et.al, Urban street tree biodiversity and antidepressant prescriptions](https://www.nature.com/articles/s41598-020-79924-5#citeas)). But what about the \
                physical health? Obviously, due to their carbon-storing \
                nature through photosynthesis, trees should have a positive effect \
                on the air-quality in a city. But is this influence trackable \
                and further more big enough to have an actual impact on the \
                health of the cities inhabitants? Or are social factors more important than city trees?")
    st.markdown("**Let's find out together!** Taking the city of New York as an example\
                we are going explore the health of the citizens, tree distribution and \
                    air pollution and see how and if they are connected.")

with right_column1:
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')
    st.markdown("###### Look at the tree distribution of NY!")
    folium_static(ny_map_trees(tree_data[2]))
    # Notify the reader that the data was successfully loaded.
    data_load_state.text("")

data_load_state = st.text('Loading...Have some patients - we are handling a lot of data!')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
st.header("Data Exploration")

st.markdown("For the analysis we used three different data sets. The dataset that we used to obtain the \
            information about the health of the citizens comes from joints data from various American Services\
            It containts information on the health status as well as \
            social status of the citizens of New York City, grouped into the districts of the city. \
            Within this analysis, the used data results from the \
            [New York City Community Health Survey (CHS)](https://www1.nyc.gov/site/doh/data/data-sets/community-health-survey.page) \
            and the [American Community Survey (ACS)](https://www.census.gov/programs-surveys/acs). \
            These surveys are conducted annually by the DOHMH, Division of Epidemiology, \
            Bureau of Epidemiology Services and U.S. Census Bureau. The CHS is a telephone survey, \
            that anually collects random samples of 10000 adults spread across the city of New York. \
            It provides a robust estimation on the health of New Yorkers, including neighborhood, \
            borough, and citywide estimates on a broad range of chronic diseases and behavioral \
            risk factors. The ACS is a nation wide Survey, that focusses on social aspects and \
            wealth, that anually collects data from about 3.5 million addresses across the \
            United States of America.",
            unsafe_allow_html=False)
st.markdown("The tree data come in three files for 1995, 2005, and 2015. \
            It describes each tree as observed by either voluanters or staff, collected by the \
            [New York City Department of Parks & Recreation](https://www.nycgovparks.org/)\
            It includes trees' diameters, heights, and status (health, dead, alive etc)\
            as well as the exact location for each tree, with lang and long, \
            which are also converted to the cartesian plane (X and Y).\
            For our project, we limit our analysis to trees that are NOT dead. \
            For geo analysis, we use the community districts of New York to explore trends. Furthermore, \
            we added park data to the dataset, to represent the green areas of the city. This park data\
            also comes from the [New York City Department of Parks & Recreation](https://www.nycgovparks.org/)\
            in form of a geo-file that states the zipcodes that the park stretches out on and the size on the park\
            in acres",
            unsafe_allow_html=False)
st.markdown("The dataset we used to analyze the airquality comes from the Environment & Health Data\
            Portal, which is run by the NYC Environmental Public Health Tracking Program and contains\
            [New York City air quality surveillance data](http://nyc.gov/health/environmentdata).",
            unsafe_allow_html=False)
    
      
            
#Options for the reader to look at the datasets
st.success("Please choose one dataset at a time, to explore the data that we used for the analysis!")
left_column2, middle_column2, right_column2 ,very_right_column2= st.columns((1,1,1,1))
with left_column2:
    health_check= st.checkbox('Explore health data')
with middle_column2:
    tree_check=st.checkbox('Explore tree data')
        
with right_column2:
    air_check=st.checkbox('Explore airquality data')
with very_right_column2:
    assumpts_check=st.checkbox('The truth: Assumptions')
        
        
if health_check and not tree_check and not air_check and not assumpts_check:
    left_column_healthdata,middle_column_healthdata, right_column_healthdata ,very_right_column_healthdata= st.columns((7,1,1,1))
    with st.spinner('Wait for it...'):
        with left_column_healthdata:
            st.markdown("#### Health dataset description")
            st.markdown("The health dataset includes information about social and health factors\
                        of the population, grouped by the 59 community districts of the city. Of the\
                        194 initial variables, 8 are chosen to be included into the analysation of this dataset. \
                        These include the socio-health factors **avoidable asthma**, **unemployment**, \
                        **self-reported health**, **smoking**, and **life expectancy**.")
            with st.expander('See details of health dataset'):
                st.markdown('''
                            Shape of initial dataset:  (68, 194)\
                            Shape of subset of dataset used for analysis: (59, 8)
                            
                            Variables in the dataset:
                            - **ID** (number): Identifier of the neighborhood geographic area, used for joining to mapping geography files to make thematic maps
                            - **Name** (text): Name of the neighborhood geographic area
                            - **Avoidable_Asthma** (number): Age-adjusted avoidable hospitalizations for asthma per 100,000 adults (AHRQ PQIs 5 and 15) (Individuals may report a homeless shelter, a post office box, the Post Office’s General Delivery Window or a residential treatment facility as their residential address. This may lead to higher hospitalization rates in some neighborhoods, depending on what institutions or residential facilities are located in the Community District)
                            - **Poverty** (number): Percent of individuals living below the federal poverty threshold
                            - **Unemployment** (number): Percent of the civilian population 16 years and older that is unemployed
                            
                            - **Self_rep_health** (number): Age-adjusted percent of adults reporting that their health is “excellent,” "very good," or “good” on a 5-level scale (Poor, Fair, Good, Very Good or Excellent)
                            - **Smoking** (number): Identifier of the neighborhood geographic area, used for joining to mapping geography files to make thematic maps
                            - **Life_expectancy_rate** (number): Life expectancy at birth in years 
                            ''')
        
                st.write(health_data)
            st.markdown("To get a feeling for the distribution of the data, in a first step the density distribution \
                        across the city is plotted.")
            st.pyplot(density_health(health_data),use_container_width=True)
            st.markdown('''
                        It looks like there is a significant difference in health and social aspects amongst the different\
                        districts of the city.
                        
                        Given that the dataset already grouped all individual data into the districts of the city\
                        we can make use of that and look at the distribution of the different factors across the city.\
                        Use the dropdown menu to the left to choose between the different variables.\
                        ''')
            # Create a text element and let the reader know the data is loading.
            data_load_state = st.text('Loading data...')
            st.plotly_chart(health_data_on_map(health_data,geodata),use_container_width=True)
            # Notify the reader that the data was successfully loaded.
            data_load_state.text("")
            st.markdown("To see if there are possible correlations between the different variables, in a next step\
                        we have a look at the four poorest and four richest districts, and how they are ranked in \
                        relation.")
            st.pyplot(poverty_ranking(health_data),use_container_width=True)
            st.markdown("This visualization allows us to see possible connections. For example are 3/4 \
                        of the poorest districts (the districts with the highest poverty)\
                        also the districts with the highest unemployment rate - that makes\
                        sense! Furthermore, we can see that the poor districts report to feel less healthy, which can\
                        also be seen in the life-expectancy, that is lower for people in the poorer districts.\
                        Interesting is also, that the asthma rate is higher for the poorer districts. Looking at the \
                        linear correlation in pairplots between the avoidable asthma rate and all other variables and the life \
                        expectancy and all other variables, reinforces the assumption that health and social status \
                        seem highly correlated.")
            st.plotly_chart(correlation_two_var(health_data),use_container_width=True)
            with st.expander('Look all the pairplots to see the correlation matrix'): 
                st.pyplot(pairplot_health(health_data),use_container_width=True)
            st.markdown("This dataset shows us that there seems to be a large correlation between the social status\
                        of a person and their health. Possibly this relationship might be more important than the \
                        airquality or tree density? Let's keep that in mind while we continue with the analysis!")
elif tree_check and not health_check and not air_check and not assumpts_check:
    left_column_treedata, middle_column_treedata, right_column_treedata, very_right_column_treedata= st.columns((1,7,1,1))
    with st.spinner('Wait for it...'):
        with middle_column_treedata:
            st.markdown("#### Tree dataset description")
            st.markdown("The tree dataset includes information about the street trees in the city\
                        of New York. This does not inlcude park trees, but trees that grow alongside roads.\
                        In the dataset there is one row for each tree, showing the position, health,\
                        kind of tree and other tree related information. The available data contains the \
                        tree count from the years 1995, 2005 and 2015. Have a look at the first 50 rows of the dataset\
                        to understand, how the dataset is built up.")
            with st.expander('Show tree dataset'):
                chosen = st.radio(
                    'Tree data from',
                    ("1995", "2005", "2015"))
                if chosen=="1995":
                    st.subheader('Tree data set from 1995')
                    st.write(pd.concat([tree_data[0].head(50)]))
                if chosen=="2005":
                    st.subheader('Tree data set from 2005')
                    st.write(pd.concat([tree_data[1].head(50)]))
                if chosen=="2015":
                    st.subheader('Tree data set from 2015')
                    st.write(pd.concat([tree_data[2].head(50)]))
            st.markdown("To get a better feeling on how the trees are distributed across the city, and to be \
                        able to later compare this data to the health and airpollution dataset, we have grouped the trees according to \
                        the different districts.")
            # Create a text element and let the reader know the data is loading.
            data_load_state = st.text('Loading data...')
            st.plotly_chart(tree_count(tree_dens_data,geodata),use_container_width=True)
            # Notify the reader that the data was successfully loaded.
            data_load_state.text("")
            st.markdown("One can see that the amount of trees had changed over the years. Let's look\
                        at the more recent change a bit more in detail.")
            st.plotly_chart(tree_change(tree_data, geodata),use_container_width=True)
            st.markdown("Great, it looks like between 2005 and 2015 the NY city council planted\
                        more trees across the city! But there are also several red areas, that \
                        indicate a loss of trees. Eventhough we will not analyse temporal patterns \
                        in this analysis, this is still interesting to have in mind.")
            with st.expander('Check out how much the tree density exactly changed per district'):
                st.pyplot(tree_change_per_district(tree_dens_data))
                st.markdown("Only 7 districts have experienced a decrease in tree density from 2005 \
                            to 2015. 4 of which are located in Queens, 2 in Brooklyn, and 1 in Bronx.\
                            Note however that by this metrics, it is more difficult to increase the \
                            trees/km2 for large areas: more trees need to be planted to have a larger \
                            number in bigger areas. Nonetheless, the tree count per unit area is a \
                            standard measure in environmental science and it is pretty descriptive, \
                            hence why it is used here.")
elif air_check and not tree_check and not health_check and not assumpts_check:
    
    left_column_treedata, middle_column_treedata, right_column_treedata, very_right_column_treedata = st.columns((1,1,7,1))
    with st.spinner('Wait for it...'):
        with right_column_treedata:
            st.markdown("#### Air quality dataset description")
            st.markdown("The air quality dataset includes information about multiple pollutants across the city\
                        of New York from 2006 to 2016. Air pollution is one of the most important \
                        environmental threats to urban populations and while all people are exposed, \
                        pollutant emissions, levels of exposure, and population vulnerability vary \
                        across neighborhoods. Exposures to common air pollutants have been linked \
                        to respiratory and cardiovascular diseases, cancers, and premature deaths. \
                        These indicators provide a perspective across time and NYC geographies to \
                        better characterize air quality and health in NYC.")
            with st.expander('Show airquality dataset'):
                    st.markdown('''
                                Variables in the dataset:
                                - **unique_id** (text): Unique record identifier
                                - **indicator_id** (number): Identifier of the type of measured value across time and space
                                - **name** (text): Name of the indicator
                                - **measure** (text): How the indicator is measured
                                - **measure_info** (text): Information (such as units) about the measure
                                - **geo_type_name** (text): Geography type; UHF' stands for United Hospital Fund neighborhoods; For instance, Citywide, Borough, and Community Districts are different geography types
                                - **geo_join_id** (text): Identifier of the neighborhood geographic area, used for joining to mapping geography files to make thematic maps
                                - **geo_place_name** (text): Neighborhood name
                                - **time_period** (text): Description of the time that the data applies to ; Could be a year, range of years, or season for example
                                - **start_date** (floating_timestamp): Date value for the start of the time_period; Always a date value; could be useful for plotting a time series
                                - **data_value** (number): The actual data value for this indicator, measure, place, and time
                                - **message** (text): notes that apply to the data value; For example, if an estimate is based on small numbers we will detail here. In this case they are all NaN
                                                            ''')
                    st.write(airquality_data)
            st.markdown("Because of the large amout of data, at first a check was made if there is an \
                        equal amount of data in all districts. The analysis shows that the data is not \
                        equally distributed. The reason for that is that for some years data has not \
                        been collected in some of the districts.")
            # Create a text element and let the reader know the data is loading.
            with st.expander('Data distribution'):
                data_load_state = st.text('Loading data...')
                st.pyplot(plot_air_equal(airquality_data))
                # Notify the reader that the data was successfully loaded.
                data_load_state.text("")
            st.markdown("Next, we want to look at the distribution of fine particle matter and sulfur\
                        dioxide concentration across the city, \
                        as this will be the data that we will compare with the other datasets.")
            st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
            air_viz1=st.radio(label = 'Choose data to visualize geographically', options = ['Fine Particle Matter','Sulfur Dioxide'])
            data_load_state = st.text('Loading data...')
            if air_viz1=='Sulfur Dioxide':
                st.plotly_chart(sulfur_on_map(airquality_data, geodata),use_container_width=True)
            else:
                st.plotly_chart(fine_particle_matter_on_map(airquality_data, geodata),use_container_width=True)
            data_load_state.text("")
            st.markdown("It seems like there is quite a big difference between the different \
                        districts of the city. It will be interesting to see, if this is connected to \
                        the tree density and if it has an influence on the health of the population!")
            st.markdown("Finally we want to have a look at the development of the two parameters \
                        over time in each district.")
            air_viz2=st.radio(label = 'Choose data for temporal visualization', options = ['Fine Particle Matter','Sulfur Dioxide'])
            data_load_state = st.text('Loading data...')
            if air_viz2=='Sulfur Dioxide':
                st.bokeh_chart(sulfur_over_time(airquality_data),use_container_width=True)
            else:
                st.bokeh_chart(fine_particle_over_time(airquality_data),use_container_width=True)
            data_load_state.text("")
            st.markdown("Well, that's good news! The airpollution has gone down over time in all \
                        districts of the city.")
                    
elif assumpts_check and not tree_check and not health_check and not air_check:
    left_column_assumpdata, middle_column_assumpdata, right_column_assumpdata, very_right_column_assumpdata = st.columns((1,1,1,7))
    with st.spinner('Wait for it...'):
        with very_right_column_assumpdata:
            st.markdown("#### Assumptions or 'the truth'")
            st.markdown("However nice this might sound so far: we just look at some NYC data, create some code \
                        and visualizations and - Abracadabra - we know if health is influenced by trees and parks\
                        in the city. The truth is that as every statistical and machine learning model, this analysis is just a perception\
                        of the real world, because the real world is a very complex place and it \
                        is impossible to model it without any assumptions. Thus, in order to model it and \
                        trying to make predictions, one needs to simplify and capture the problem with numbers \
                        and assumptions. The numbers that our analysis is \
                        based on can be found and explored in the datasets, so here it's time to get down to \
                        reality and outline the main assumptions and limitations of our analysis.")
            st.markdown('''
                        1.) The available data was collected at different years. While the tree and airquality\
                            dataset are from 2015, the health data was collected between 2011 and 2013 and then \
                            averaged. Thus, our first assumption is that **the health and social class\
                            of the people of NYC has not changed significantly between the different districts \
                            of the city between 2013 and 2015**. 
                            
                        2.) To calculate the tree density and tree count all trees are treated equally and no difference\
                            is made between healthy or ill, big or small trees. Our second assumption is that \
                            **all trees have the same influence.**
                            
                        3.) For the park data, parks that stretched over more than one community district were \
                            distributed equally to these districts, assuming that **big parks distribute equally\
                            across all districts they are part of.** Furthermore, we simplified that **all parks are the same**. As we did not differentiate between playgrounds, running fields and \
                            forests.
                            
                        4.) Finally we assume that **every district is homogenous across itself**. \
                            Thus it is sufficient to group the data into the different communal districts for the analysis.
                        ''')
                    
st.markdown("Overall, the data exploration shows that there are big differences in regards to all variables across the\
            city of New York. There are areas with high airpollution like **Sheepheads Bay** and areas with lower airpollution like **East New York and Starrett City**, with a big \
            tree density like **Upper West Side** and a low tree density like **Coney Island** and wealthy areas like **Tottenville and Great Kills** and poor areas like **Morrisania and Crotona**. \
            ")
c1,c2,c3=st.columns((1,2,1))
with c2:
    st.bokeh_chart(density_all(airquality_data,health_data,tree_dens_data),use_container_width=True)

st.markdown("When looking at the density distribution of each variable across the city, these findings are confirmed.\
                The distributions differ a lot between the different variables and no pattern is visible at first sight.\
                Also the shape of the distributions changes quite significantly and does mainly not follow a nice \
                normal distribution.")
        
data_load_state.text("")


#%%%%%%%%%%%%%%%%%%%%%%% Correlation Analysis 
st.header("Correlation Analysis between airquality, health and tree density")

            
st.markdown("Now that we know a bit about the data and the assumptions that we \
                made for the analysis, we want to move on to the more interesting part that\
                comes closer to hopefully answering our question, if we will actually\
                be able to see an effect of trees and parks in the city on the physical health of the citizens\
                - the correlation between the datasets.")
st.markdown("To analyze the correlation between the different datasets, we defined important variables from each\
                dataset, that were relevant for our analysis.\
                Then we merged them together into one dataset, grouped by the unique community districts. \
                ")

    #st.markdown('''#### $R^2$ Value \
    #            
    #            Another indicator of correlation is the $R^2$ Value. The R2 value indicates how much a \
    #            change in one variable influences the change on another variable. In this analysis this \
    #            is tested independent for each pair of variables.\
    #            The pairwise $R^2$ values for the listed variables are listed below:\
    #            ''')
    #st.pyplot(r_square(airquality_data,health_data,tree_dens_data))


st.markdown('''
                ### Correlation Matrix
                The correlation matrix shows which variables are correlated and which not,\
                claculated with the Pearson coefficient, a linear measure\
                of the correlation  between each of the variables. Big blue boxes symbolize a high positive correlation,
                meaning that when one of the pairs' values increase, the other pairs' values increase too. And big\
                red boxes symbolize a high negative correlation, meaning that when one of the pairs' values increase, \
                the other pairs' values decrease.''')
cl1,cl2,cl3=st.columns((1,3,1))
with cl2:
    st.pyplot(corrplot(airquality_data, health_data, tree_dens_data))
    
cll1,cll2=st.columns((2,2))
with cll1:
    with st.expander("The corresponding Poisson values"):
        st.pyplot(air_health_tree_correlation(airquality_data,health_data,tree_dens_data))
with cll2:
    with st.expander("See the math behind it"):
        st.markdown("The Pearson correlation coefficient is calculated in the following way:")
        
        st.image(correlation_im)
        st.markdown('''
                    With n being the sample size (in our case the number of districts), \
                    $x_i,y_i$ the individual sample points (in our case the value of the different \
                    parameters in our data for each district). 
                        
                    You can find more information [here](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient).
                    ''')
        
st.markdown("One can see that the main correlation is found between the social and health variables. \
            That indicates that, when viewed independently, the variables from the different datasets don't \
            have a high influence on the variance of one another and thus the amount of street trees \
            does not seem to directly influence the physical health of the citizens. However, there seems\
            to be a almost shocking high correlation between the social factors, especially `Poverty` and `Unemployment`\
            and the health and life expectancy of people. Rich people seem to live a lot longer than poor people and \
            suffer from avoidable asthma significantly less.\
            Look at the following pairplot, so see with the help of the pairwise linear regression of the \
            analyzed variables, how strong the linear correlation between social factors and health is.")
with st.expander("Pairplot of the analyzed variables"):
    st.pyplot(pairplots(airquality_data,health_data,tree_dens_data))
    


#%%%%%%%%%%%%%%%%%%%%%%% Machine Learning
st.header("Machine Learning model")

st.write("We create two models that use data related to the various \
districts of New York to predict the Health related rates among the population.")

with st.expander("Variables used in the model"):
    st.write("The data related to each district used in the training consists of the following:  \n"\
    """    
    * Tree counts  \n \
    * District areas  \n \
    * Trees' density  \n \
    * Percentage district area occupied by parks  \n \
    * Fine Particulate Matter (PM2.5) (mcg per cubic meter)  \n \
    * Sulfur Dioxide (SO2) (ppb)  \n \
    * Poverty rate  \n \
    * Smoking rate  \n \
    * Unemployment rate  \n \
    * Self reported health status
    """)
st.write("The models will try to predict the following variables:  \n"\
"""    
* Number of people with Avoidable Asthma per 100 000 people  \n \
* Life expectancy \n \
""")


# Loading the df used for the models, preprocessing included
df_final = merge_data(airquality_data, health_data, tree_dens_data) # final df used to build the model

st.markdown("#### Model creation and prediction")
st.write("The two models are created using xgboost. The columns used for training the model are \
  those indicated above, while the target variable are the Avoidable Asthma Rate and the \
Life Expactancy Rate among the various district for the first and second model respectively. \
The comparison between the real data and the predicted data is presented below. ")
	# Creation of model 1
xgb_model_1, y_test_1, y_pred_1, mse_1 = training_model_1(df_final)
	# Creation of model 2
xgb_model_2, y_test_2, y_pred_2, mse_2 = training_model_2(df_final)
	
st.bokeh_chart(plot_model_results(df_final, xgb_model_1, y_test_1, y_pred_1, mse_1, xgb_model_2, y_test_2, y_pred_2, mse_2),use_container_width=True)
	
st.write("Results: As we can see from the graph, the model manages to predict the general \
 behaviour of the health rates analyzed.  \n")

st.markdown("#### Model interpretation")
st.write("We want now to analyze the importance of each variable used to predict the two health \
         factors. Which one is more important? Tree density or Poverty rate, Smoking rate or \
            Sulfur Dioxide levels in the district?")
	# plot model interpretability
st.pyplot(model_interpretation(df_final, xgb_model_1, xgb_model_2),use_container_width=True)
st.write("As we can see from the graph above, the most important features calculated from the \
        prediction model are the social aspects of the population. Unemployment and Poverty \
        appear to be the most important variables in avoiding Asthma and extending the Life Expectancy. \
        The latter, appears to be also determined by the Fine particular matter (PM2.5) level in \
        the districts. Trees density appear to have very low effect in predicting the health of \
        the population.")
	
#%%%%%%%%%%%%%%%%%%%%%%%Conclusion
st.header("Conclusion")
st.markdown('''The purpose of this analysis was to understand what is the relation between health, \
trees and air quality among the districts of New York City. We studied different datasets, \
and built a Machine Learning Model that predicted Avoidable Asthma Rate and Life Expectancy \
Rate in 2015 using social aspects, air quality levels and tree density in the various districts. \
The results are that the most important variables when considering health are social factors \
related to the economic conditions of the people studied, in our case employment and poverty. \
Air pollution also affects the Life expectancy, although we only found low influence when \
considering the Sulfur Dioxide levels, probably because of the low variance between district \
in the chosen year. On the other hand, the amount of trees in a districts appears to have very \
low importance. A reason might be that probably the effect of trees on health can only \
be seen in a greater scale, and not analyzing each specific district. Furthermore, the \
trees that we look at are trees that are planted next to streets. Thus more trees also indicates\
more streets.

Overall, although we can't see a direct correlation between the number of trees and the physical \
health of the citizens, trees are an important factor of life in a city.
Yet, in order to predict the physical health of the citizens of NY one has to take \
into account a lot more factors than just the number of trees in the districts, most of all the\
social factors. 
''')

#%%%% Hide Menu
hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden; }
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)
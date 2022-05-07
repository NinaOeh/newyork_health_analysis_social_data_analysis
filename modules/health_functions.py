# -*- coding: utf-8 -*-
"""
Created on Tue May  3 13:43:43 2022

@author: ninao
"""

import pandas as pd
import numpy as np
import folium
from matplotlib import pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import scipy

def health_data_on_map(health_data,geodata):
    data1=[]
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Poverty'],
                                        colorbar=dict(title='Poverty <br>(% below minimum)'),
                                        colorscale="sunset"
                                        ))
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Avoidable_Asthma'],
                                        colorbar=dict(title='Avoidable <br> Asthma <br> (nr. persons)'),
                                        colorscale="sunset"
                                        ))
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Self_rep_health'],
                                        colorbar=dict(title='Self reported <br> health (%)'),
                                        colorscale="sunset"
                                        ))
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Smoking'],
                                        colorbar=dict(title='Smoking (%)'),
                                        colorscale="sunset"
                                        ))
    data1.append(go.Choroplethmapbox(geojson=geodata, 
                                        locations=health_data["ID"], 
                                        z=health_data['Life_expectancy_rate'],
                                        colorbar=dict(title='Life expectancy <br>(age)'),
                                        colorscale="sunset"
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
    health_data["Self_rep_health_rank"]=health_data["Self_rep_health"].rank(method='max')
    health_data["Smoking_rank"]=health_data["Smoking"].rank(method='max')
    health_data["Avoidable_Asthma_rank"]=health_data["Avoidable_Asthma"].rank(method='max')
    health_data["Life_expectancy_rate_rank"]=health_data["Life_expectancy_rate"].rank(method='max')
    
    
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
    data.append(go.Scatter(name="Poverty",x=health_data["Avoidable_Asthma_norm"], y=health_data["Poverty_norm"],mode="markers",marker={'color':'blue'}))
    data.append(go.Scatter(name="Unemployment",x=health_data["Avoidable_Asthma_norm"], y=health_data["Unemployment_norm"],mode="markers",marker={'color':'red'}))
    data.append(go.Scatter(name="Self reported health",x=health_data["Avoidable_Asthma_norm"], y=health_data["Self_rep_health_norm"],mode="markers",marker={'color':'green'}))
    data.append(go.Scatter(name="Smoking",x=health_data["Avoidable_Asthma_norm"], y=health_data["Smoking_norm"],mode="markers",marker={'color':'orange'}))
    data.append(go.Scatter(name="Life expectancy",x=health_data["Avoidable_Asthma_norm"], y=health_data["Life_expectancy_rate_norm"],mode="markers",marker={'color':'purple'}))
    
    data.append(go.Scatter(x=np.array(health_data["Avoidable_Asthma_norm"]), y=regression_line(np.array(health_data["Avoidable_Asthma_norm"]),np.array(health_data["Poverty_norm"])),mode="lines",showlegend=False,line= dict(color="blue")))
    data.append(go.Scatter(x=np.array(health_data["Avoidable_Asthma_norm"]), y=regression_line(np.array(health_data["Avoidable_Asthma_norm"]),np.array(health_data["Unemployment_norm"])),mode="lines",showlegend=False,line= dict(color="red")))
    data.append(go.Scatter(x=np.array(health_data["Avoidable_Asthma_norm"]), y=regression_line(np.array(health_data["Avoidable_Asthma_norm"]),np.array(health_data["Self_rep_health_norm"])),mode="lines",showlegend=False,line= dict(color="green")))
    data.append(go.Scatter(x=np.array(health_data["Avoidable_Asthma_norm"]), y=regression_line(np.array(health_data["Avoidable_Asthma_norm"]),np.array(health_data["Smoking_norm"])),mode="lines",showlegend=False,line= dict(color="orange")))
    data.append(go.Scatter(x=np.array(health_data["Avoidable_Asthma_norm"]), y=regression_line(np.array(health_data["Avoidable_Asthma_norm"]),np.array(health_data["Life_expectancy_rate_norm"])),mode="lines",showlegend=False,line= dict(color="purple")))
    
    #life expectancy on x axis
    data.append(go.Scatter(name="Poverty",x=health_data["Life_expectancy_rate_norm"], y=health_data["Poverty_norm"],mode="markers",marker={'color':'blue'}))
    data.append(go.Scatter(name="Unemployment",x=health_data["Life_expectancy_rate_norm"], y=health_data["Unemployment_norm"],mode="markers",marker={'color':'red'}))
    data.append(go.Scatter(name="Self reported health",x=health_data["Life_expectancy_rate_norm"], y=health_data["Self_rep_health_norm"],mode="markers",marker={'color':'green'}))
    data.append(go.Scatter(name="Smoking",x=health_data["Life_expectancy_rate_norm"], y=health_data["Smoking_norm"],mode="markers",marker={'color':'orange'}))
    data.append(go.Scatter(name="Avoidable asthma",x=health_data["Life_expectancy_rate_norm"], y=health_data["Avoidable_Asthma_norm"],mode="markers",marker={'color':'purple'}))
    
    data.append(go.Scatter(x=np.array(health_data["Life_expectancy_rate_norm"]), y=regression_line(np.array(health_data["Life_expectancy_rate_norm"]),np.array(health_data["Poverty_norm"])),mode="lines",showlegend=False,line= dict(color="blue")))
    data.append(go.Scatter(x=np.array(health_data["Life_expectancy_rate_norm"]), y=regression_line(np.array(health_data["Life_expectancy_rate_norm"]),np.array(health_data["Unemployment_norm"])),mode="lines",showlegend=False,line= dict(color="red")))
    data.append(go.Scatter(x=np.array(health_data["Life_expectancy_rate_norm"]), y=regression_line(np.array(health_data["Life_expectancy_rate_norm"]),np.array(health_data["Self_rep_health_norm"])),mode="lines",showlegend=False,line= dict(color="green")))
    data.append(go.Scatter(x=np.array(health_data["Life_expectancy_rate_norm"]), y=regression_line(np.array(health_data["Life_expectancy_rate_norm"]),np.array(health_data["Smoking_norm"])),mode="lines",showlegend=False,line= dict(color="orange")))
    data.append(go.Scatter(x=np.array(health_data["Life_expectancy_rate_norm"]), y=regression_line(np.array(health_data["Life_expectancy_rate_norm"]),np.array(health_data["Avoidable_Asthma_norm"])),mode="lines",showlegend=False,line= dict(color="purple")))
    
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
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:52:00 2022

@author: ninao
"""
#All needed imports

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import layout
from bokeh.models.widgets import Tabs, Panel
from bokeh.io import curdoc


import plotly.graph_objects as go


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
                                        colorscale="YlGn"
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
                                        colorscale="YlGn"
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
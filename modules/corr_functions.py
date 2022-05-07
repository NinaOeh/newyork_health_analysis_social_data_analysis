# -*- coding: utf-8 -*-
"""
Created on Fri May  6 16:47:50 2022

@author: ninao
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



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
    df_air_trees_health = pd.concat([df_trees['counts_per_km2_2015'],df_trees['park_percent'], df_air_col_1, df_air_col_2,df_health[["Avoidable_Asthma","Poverty","Unemployment","Self_rep_health","Smoking","Life_expectancy_rate"]]], axis=1, join="inner")
    #df_air_trees_health['Geo Place Name'] = pd.Series(id_neighbourhood_dict)
    df_air_trees_health=df_air_trees_health.rename(columns={"counts_per_km2_2015":"Tree count","park_percent":"Park percentage","Self_rep_health":"Self reported health","Life_expectancy_rate":"Life expectancy","Avoidable_Asthma":"Avoidable Asthma"})
    return df_air_trees_health

def air_health_tree_correlation(df_air,df_health,df_trees):
    
    df_air_trees_health=merge_data(df_air,df_health,df_trees)
    
    #correlation plot
    corr = df_air_trees_health.corr()
    ax = sns.heatmap(corr,vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(250, 15, s=75, l=40, n=100), square=True,annot=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90,horizontalalignment='right')
    ax.set_title('Correlation matrix of Air vs Tree vs Health dataframe')
    
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
    data=merge_data(df_air,df_health,df_trees).corr()
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
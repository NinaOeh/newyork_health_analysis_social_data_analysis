# -*- coding: utf-8 -*-
"""
Created on Tue May  3 17:51:51 2022

@author: ninao
"""
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.graph_objects as go



#Tree Dataset
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
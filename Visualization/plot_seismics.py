# -------------------------------------------------------------------------------------------------------------------------------------------------------------
# Title:  plot_seismics.py 
# Author(s):  
# Initial version:  2024-01-15
# Last modified:    2024-01-15              
# This version:     2024-01-15
# --------------------------------------------------------------------------------------------------------------------------------------------------------------
# Objective(s): To implement the functions required to produce visualizations similar to those appearing in the paper. 
#               This includes maps where the epicenter and the reacting stations are shown, as well as time series displaying 
#               the time periods of major reactions.
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Library dependencies: geopandas, contextily, matplotlib, xyzservices, pandas, numpy, pygsp, scipy, matplotlib, shapely, Models.aux_functions. 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------
# Keywords: Seismics, New Zealand, epicenter
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------

import geopandas
import contextily as cx
import matplotlib.pyplot as plt
import xyzservices.providers as xyz
import pandas as pd
import numpy as np
from pygsp import graphs
import scipy.io
import matplotlib.pyplot as plt
import pickle
import itertools
from shapely.geometry import LineString
import matplotlib.cm as cm
import matplotlib

import sys
import os

# Append the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.aux_functions import *


def plot_graph(file_name,G,affected_nodes,epicenter):
    ###### This function generates a plot of the graph related to the event under analysis.

    ##### Input:
    ## file_name: The name of the file where the results will be stored.
    ## G: the graph representing the stations.
    ## affected_nodes: the list of nodes that are affected by the event.
    ## epicenter: the geolocation of the event.


    coordinates = pd.DataFrame(G.coords[:,[1,0]], columns=['Longitude','Latitude'])

    ########## Positions of node
    gdf = geopandas.GeoDataFrame(
        coordinates, geometry=geopandas.points_from_xy(coordinates.Longitude,coordinates.Latitude), crs="EPSG:4326"
        )

    ########### Epicenter
    epicenter=pd.DataFrame(epicenter.reshape((1,2)),columns=["Latitude","Longitude"])
    epicenter = geopandas.GeoDataFrame(
    epicenter, geometry=geopandas.points_from_xy(epicenter.Longitude,epicenter.Latitude), crs="EPSG:4326"
    )

    ########## Afected_nodes
    affected_nodes = geopandas.GeoDataFrame(
    coordinates.iloc[affected_nodes], geometry=geopandas.points_from_xy(coordinates.iloc[affected_nodes].Longitude,
                                                                        coordinates.iloc[affected_nodes].Latitude), crs="EPSG:4326"
    )


    #### Network 

    ki, kj = np.nonzero(G.A)

    coordinates_1=G.coords[ki,:]
    coordinates_1= pd.DataFrame(coordinates_1[:,[1,0]], columns=['Longitude','Latitude'])
    coordinates_1 = geopandas.GeoDataFrame(
    coordinates_1, geometry=geopandas.points_from_xy(coordinates_1.Longitude,coordinates_1.Latitude),
        crs="EPSG:4326"
        )

    coordinates_2=G.coords[kj,:]
    coordinates_2= pd.DataFrame(coordinates_2[:,[1,0]], columns=['Longitude','Latitude'])
    coordinates_2 = geopandas.GeoDataFrame(
    coordinates_2, geometry=geopandas.points_from_xy(coordinates_2.Longitude,coordinates_2.Latitude),
        crs="EPSG:4326"
        )

    coordinates_1 = coordinates_1.to_crs(epsg=3857)
    coordinates_2=coordinates_2.to_crs(epsg=3857)

    geom1 = coordinates_1.values
    geom2 =coordinates_2.values
    # Cartesian product
    geom = []
    for i in range(len(ki)):
        geom.append(LineString([geom1[i][0:2],geom2[i][0:2]]))
    result= geopandas.GeoDataFrame({'geometry':geom})


    ################# Plots

    gdf = gdf.to_crs(epsg=3857)
    epicenter=epicenter.to_crs(epsg=3857)
    affected_nodes=affected_nodes.to_crs(epsg=3857)
    result.crs="EPSG:4326"
    result=result.to_crs(epsg=3857)

    ax=result.plot(figsize=(10, 10),color="black",aspect=1,alpha=0.5)
    gdf.plot(alpha=0.9, markersize=200,ax=ax,color="purple")
    affected_nodes.plot(color='red', markersize=200,ax=ax)
    epicenter.plot(ax=ax,marker='*', color='red', markersize=800)
    cx.add_basemap(ax,source=cx.providers.OpenTopoMap,attribution_size=6)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    plt.savefig(file_name+"_network.pdf")



def update_colors_green_red(ax):
    
    ##### This functions assigns a different color to the time series if they refer before and after the change points
    ### Input
    # ax: the axis of the plots
    
    cmap_green = matplotlib.colormaps["Greens"]
    cmap_reed = matplotlib.colormaps["Reds"]
    lines = ax.lines
    colors = cmap_green(np.linspace(0.3, 0.8, len(lines)))
    for line, c in zip(lines, colors):
        line.set_color(c)
        
    lines = ax.lines[3:]
    colors = cmap_reed(np.linspace(0.3, 0.8, len(lines)))
    for line, c in zip(lines, colors):
         line.set_color(c)
    
        
def update_colors_red(ax):
    
    ##### This functions assigns a different color to the time series if they refer before and after the change points
    ### Input
    # ax: the axis of the plots
    
    lines = ax.lines[3:]
    cmap_reed = matplotlib.colormaps["Reds"]
    colors = cmap_reed(np.linspace(0.3, 0.8, len(lines)))
    for line, c in zip(lines, colors):
        line.set_color(c)
        
        
def plot_results_seismic(file_name,complete_data_ref,complete_data_test,G,epicenter,sample_size_node,n_times,n_nodes,p_values_1,p_values_2=None,p_value=0.05,min_size_cluster="max"):
    #### This function replicates the plots provided in the paper.

    ### Input:
    # file_name: the name of the image file.
    # complete_data_ref: preprocessed dataset before the earthquake.
    # complete_data_test: preprocessed dataset after the earthquake.
    # G: the spatial graph made of the stations.
    # epicenter: the geolocation of the event.
    # sample_size_node: the number of observations per window.
    # n_times: number of time windows.
    # n_nodes: number of stations.
    # p_values_1: the p-values associated with the multiple two-sample tests.
    # p_values_2: the p-values associated with the multiple two-sample tests (if MMD is used, this parameter is set to None).
    # p_value: the FWER error control.
    # min_size_cluster: (int) the minimum number of nodes required to form a cluster. If set to "max", the largest cluster is used.

    if p_values_2 is None:
        x_in=p_values_1<=p_value
    else:
        p_values_min=np.minimum(p_values_1,p_values_2)
        x_in=p_values_min<=p_value/2
        
    
    W_1=transform_matrix_totime(G.W.tocoo(),n_times=n_times)

    aux_clusters=get_componentes(x_in,W_1)
    
    time_lenght=sample_size_node
    
    nodes=dict()
    times=dict()
    clusters=dict()
    n_cluster=0

    if min_size_cluster=="max" and len(aux_clusters)!=0:
        max_cluster=max([len(cluster) for cluster in aux_clusters])
        for cluster in aux_clusters:
            if len(cluster)==max_cluster:
                clusters["cluster-"+str(n_cluster)]={"node":[],"time":[]}
                for i in cluster:
                    clusters["cluster-"+str(n_cluster)]["node"].append(i%n_nodes)
                    clusters["cluster-"+str(n_cluster)]["time"].append(int(i/n_nodes))
                
                clusters["cluster-"+str(n_cluster)]["node"]=np.vstack(clusters["cluster-"+str(n_cluster)]["node"])
                clusters["cluster-"+str(n_cluster)]["time"]=np.vstack(clusters["cluster-"+str(n_cluster)]["time"])
                n_cluster+=1
    else:            
        for cluster in aux_clusters:
            if len(cluster)>=min_size_cluster:
                clusters["cluster-"+str(n_cluster)]={"node":[],"time":[]}
                for i in cluster:
                    clusters["cluster-"+str(n_cluster)]["node"].append(i%n_nodes)
                    clusters["cluster-"+str(n_cluster)]["time"].append(int(i/n_nodes))
                    
                    clusters["cluster-"+str(n_cluster)]["node"]=np.vstack(clusters["cluster-"+str(n_cluster)]["node"])
                    clusters["cluster-"+str(n_cluster)]["time"]=np.vstack(clusters["cluster-"+str(n_cluster)]["time"])
                    n_cluster+=1

    COLOR_CYCLE = ["#4286f4", "#f44174"]
      
    if n_cluster==1:
        nodes_to_plot=clusters["cluster-"+str(0)]["node"]
        nodes_to_plot=np.unique(nodes_to_plot)
        graph_signal=np.zeros(n_nodes)
        graph_signal[nodes_to_plot]=1

        plot_graph(file_name,G,nodes_to_plot,epicenter)

        cmap_green = matplotlib.colormaps["Greens"]
        cmap_reed = matplotlib.colormaps["Reds"]
        
        fig,axs = plt.subplots(len(nodes_to_plot),figsize=(30,2*len(nodes_to_plot)))

        if len(nodes_to_plot)==1:
            index_node=np.where(clusters["cluster-"+str(0)]["node"]==nodes_to_plot[0]) 
            time=clusters["cluster-"+str(0)]["time"][index_node]
            start=[time_lenght*t for t in time]
            end=[time_lenght*(t+1) for t in time]  
            alpha = 0.2  
            for t in range(len(start)):  
                axs.axvspan(max(0, start[t] - 0.5), end[t] - 0.5, facecolor=COLOR_CYCLE[t%2], alpha=alpha)
            axs.plot(complete_data_ref[nodes_to_plot[0]])
            axs.plot(complete_data_test[nodes_to_plot[0]],alpha=0.5)
            update_colors_green_red(axs)

        else:
            index_nodes=np.argsort(np.sum((G.coords-epicenter)**2,axis=1)[nodes_to_plot])
            for i in index_nodes:
                index_node=np.where(clusters["cluster-"+str(0)]["node"]==nodes_to_plot[i]) 
                time=clusters["cluster-"+str(0)]["time"][index_node]
                start=[time_lenght*t for t in time]
                end=[time_lenght*(t+1) for t in time]  
                alpha = 0.2 
                for t in range(len(start)):  
                    axs[i].axvspan(max(0, start[t] - 0.5), end[t] - 0.5, facecolor=COLOR_CYCLE[t%2], alpha=alpha)
                    
                axs[i].plot(complete_data_ref[nodes_to_plot[i]])
                axs[i].plot(complete_data_test[nodes_to_plot[i]],alpha=0.5)
                update_colors_green_red(axs[i])
                axs[i].set_yticklabels([])
                axs[i].set_xticklabels([])

        xticks=np.arange(1100,step=100)
        labels=["Os","5s","10s","15s","20s","25s","30s","35s","40s","45s","50s"]
        axs[len(nodes_to_plot)-1].set_xticks(xticks,labels,fontsize=40)
    
        plt.savefig(file_name+"_waveforms.pdf")
        
    elif n_cluster>=1:
        for c in range(len(clusters)):  
            
            file_name_=file_name+"_cluster-"+str(c)
            nodes_to_plot=clusters["cluster-"+str(c)]["node"]
            nodes_to_plot=np.unique(nodes_to_plot)
            # affected_nodes=nodes_to_plot    
            graph_signal=np.zeros(n_nodes)
            graph_signal[nodes_to_plot]=1
            axs_left = sfigs[c][0].subplots(1, 1)

            plot_graph(file_name,G,nodes_to_plot,epicenter)
            
            cmap_green = matplotlib.colormaps["Greens"]
            cmap_reed = matplotlib.colormaps["Reds"]
 
            fig,axs = plt.subplots(len(nodes_to_plot),figsize=(30,2*len(nodes_to_plot)))
            
            if len(nodes_to_plot)==1:
                index_node=np.where(clusters["cluster-"+str(c)]["node"]==nodes_to_plot[0]) 
                time=clusters["cluster-"+str(c)]["time"][index_node]
                start=[time_lenght*t for t in time]
                end=[time_lenght*(t+1) for t in time]  
                alpha = 0.2  
                for t in range(len(start)):  
                    axs.axvspan(max(0, start[t] - 0.5), end[t] - 0.5, facecolor=COLOR_CYCLE[t%2], alpha=alpha)
                axs.plot(complete_data_ref[nodes_to_plot[0]])
                axs.plot(complete_data_test[nodes_to_plot[0]],alpha=0.5)
                update_colors_green_red(axs)

            else:
                index_nodes=np.argsort(np.sum((G_seismic.coords-epicenter)**2,axis=1)[nodes_to_plot])
                for i in index_nodes:
                    index_node=np.where(clusters["cluster-"+str(c)]["node"]==nodes_to_plot[i]) 
                    time=clusters["cluster-"+str(c)]["time"][index_node]
                    start=[time_lenght*t for t in time]
                    end=[time_lenght*(t+1) for t in time]  
                    alpha = 0.2 
                    for t in range(len(start)):  
                        axs[i].axvspan(max(0, start[t] - 0.5), end[t] - 0.5, facecolor=COLOR_CYCLE[t%2], alpha=alpha)
                    axs[i].plot(complete_data_ref[nodes_to_plot[i]])
                    axs[i].plot(complete_data_test[nodes_to_plot[i]],alpha=0.5)
                    update_colors_green_red(axs[i])

            xticks=np.arange(1100,step=100)
            labels=["Os","5s","10s","15s","20s","25s","30s","35s","40s","45s","50s"]
            axs[len(nodes_to_plot)-1].set_xticks(xticks,labels,fontsize=40)
                
        plt.savefig(file_name+"_waveforms.pdf")
    else:
        with open(file_name+"_results.txt", "w") as file:
            file.write('Not clusters found')        

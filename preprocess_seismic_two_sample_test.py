# -----------------------------------------------------------------------------------------------------------------
# Title:  preprocess_seismic_two_sample_test
# Author(s):  Alejandro de la Concha
# Initial version:  2024-01-15
# Last modified:    2025-02-28              
# This version:     2025-02-28
# -----------------------------------------------------------------------------------------------------------------
# Objective(s): The goal of this script is to recover the graph structure and the waveforms associated with a given event
#               coming from a specified ID.
# -----------------------------------------------------------------------------------------------------------------
# Library dependencies:
# obspy, pandas, numpy, matplotlib, statsmodels, sklearn, scipy, pygsp
# -----------------------------------------------------------------------------------------------------------------
# Keywords: GEONET, seismics, k-nearest neighbor
# -----------------------------------------------------------------------------------------------------------------

from obspy import UTCDateTime
import obspy.signal
from obspy.clients.fdsn import Client as FDSN_Client
from obspy import read_inventory
import pandas as pd
from obspy import Stream
import numpy as np
import pickle
import copy
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.neighbors import kneighbors_graph
from scipy.spatial import distance_matrix
from pygsp import graphs
import argparse

def get_data(network,event_id,seconds_before,seconds_after):
##### This function uses Obspy to access the waveforms associated with the event.

### Input:
# network: list of available estimations.
# event_id: identifier of the event, available on the GEONET webpage.
# seconds_before: number of seconds before the event to be used in this task.
# seconds_after: number of seconds after the event to be used in this task.

### Output:
# data: a list where each element contains a 3D numpy array with the preprocessed observations.
# epicenter: the location of the epicenter.



    client = FDSN_Client("GEONET")
    cat = client.get_events(eventid=event_id)
    otime = cat[0].origins[0].time
    inventory = client.get_stations(latitude=cat[0].origins[0].latitude,
                                longitude=cat[0].origins[0].longitude,
                                maxradius=100,
                                channel="HHZ",
                                level="channel",
                                starttime = otime-seconds_before-10,
                                endtime = otime+seconds_after+10)

    epicenter=np.array((cat[0].origins[0].latitude,cat[0].origins[0].longitude))
    data={}
    print(event_id)
    sensor_name=["HHZ","HHE","HHN"]
    
    if(int(event_id[:4])>=2024):
        code=inventory[1].code
    else:
        code=inventory[0].code
        
 #   station=network[0]
  #  i=0
    for station in network:
        try:
            data[station]=[]
            for i in range(3):
                try:
                   
                    aux_waveform=client.get_waveforms(code, station, "*",sensor_name[i],
                                       otime-seconds_before-10, otime + seconds_after+10,attach_response=True)
                    
                    aux_waveform.remove_response(output="VEL")
                    aux_waveform.detrend("linear")
                    aux_waveform.detrend(type='demean')
                    aux_waveform.filter("bandpass",freqmin=2, freqmax=16)
                    down_sampled=obspy.signal.filter.integer_decimation(aux_waveform[0].data,decimation_factor=5)[199:]
                    down_sampled=down_sampled[:len(down_sampled)-200]
            
                    model = AutoReg(down_sampled, lags=1)
                    model_fit = model.fit()
                    residuals=down_sampled-model_fit.predict()
                    residuals=residuals[1:]
             
                    mean_residuals=np.mean(residuals)
                    std_residuals=np.std(residuals)
                    residuals=(residuals-mean_residuals)/std_residuals
                    data[station].append(1*residuals)
                
                except:
                    print(station)
                    print(sensor_name[i])
                    pass
             
            if len(data[station])<3:
                del data[station]
            else:
                max_index=min([len(d) for d in data[station]])
                data[station]=[d[:max_index] for d in data[station]]
                data[station]=np.vstack(data[station]).transpose()
                
        except:
            pass          
    return data,epicenter

def get_graph_waveforms(data_directory,event_id):    
##### This function generates three files: one containing all the waveforms and two others with the network between stations and their locations.

### Input:
# data_directory: the folder where the datasets will be saved.
# event_id: identifier of the event, available on the GEONET webpage.


    seconds_before=50.0
    seconds_after=50.0   
    client = FDSN_Client("GEONET")
    cat = client.get_events(eventid=event_id)
    otime = cat[0].origins[0].time
    inventory = client.get_stations(latitude=cat[0].origins[0].latitude,
                                    longitude=cat[0].origins[0].longitude,
                                    maxradius=100,
                                    channel="HHZ",
                                    level="channel",
                                    starttime = otime-seconds_before,
                                    endtime = otime+seconds_after)
    
    if(int(event_id[:4])>=2024):
        network=[station.code for station in inventory[1].stations]
    else:
        network=[station.code for station in inventory[0].stations]
        
    waveforms,epicenter=get_data(network,event_id,seconds_before,seconds_after)   
    coordinates={}
    for network in inventory:
        for i in range(len(network)):
            if network[i].longitude>0:
                if network[i].code in waveforms.keys():
                    aux_coordinates_1=network[i].latitude
                    aux_coordinates_2=network[i].longitude
                    coordinates[network[i].code]=np.array((aux_coordinates_1,aux_coordinates_2))
            
    name_stations=set( waveforms.keys()).intersection(set(coordinates.keys()))
    coordinates=np.array([coordinates[station] for station in name_stations])
    
    name_stations=list(name_stations)
    distance_matrix_=distance_matrix(coordinates,coordinates)
    distance_matrix_= kneighbors_graph(coordinates, 3, mode='connectivity', include_self=False)

    distance_matrix_=distance_matrix_.todense()
    distance_matrix_=distance_matrix_+distance_matrix_.transpose()
    distance_matrix_[distance_matrix_>0]=1

    network_sensors=pd.DataFrame(distance_matrix_,columns=name_stations,index=name_stations)
    coordinates=pd.DataFrame(coordinates,columns=["latitude","longitude"],index=name_stations)
    
    waveforms={x: v for x, v in waveforms.items() if x in name_stations}
                    
    seismic_elements={"waveforms":waveforms,"epicenter":epicenter}
    with open(data_directory+'/'+event_id+'_data.pickle', 'wb') as handle:
        pickle.dump(seismic_elements, handle, protocol=pickle.HIGHEST_PROTOCOL)    
        
    file_name="/New_Zealand_Network_"+event_id+".csv"
    network_sensors.to_csv(data_directory+file_name)

    file_name="/New_Zealand_coordinates"+event_id+".csv"
    coordinates.to_csv(data_directory+file_name)   


if __name__=="__main__":

    parser=argparse.ArgumentParser(description="Parameters to replicate experiments")
    parser.add_argument("--data_directory") #### Dictionary where the results will be saved
    parser.add_argument("--eventid",type=str) #### The name of the model to be run
    
    args=parser.parse_args()
    
    
    data_directory=args.data_directory
    event_id=args.eventid
    get_graph_waveforms(data_directory,event_id)











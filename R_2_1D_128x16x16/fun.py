import pickle
import numpy as np
import torch as t
from scipy.interpolate import Rbf


def load_deap_pkl(dirr):
    """Open pickled deap dict. Label order: valence, arousal, dominance, liking"""
    with open(dirr, 'rb') as f:
        data = pickle.load(f)
    return data


def streams_for_3dConvnet_cat(x,y):
    x_stream=t.empty((int(60*x.shape[0]),128,9,9))
    y_stream=t.empty((x_stream.shape[0]),1)
    def c(x,i):
        return x[i,:].view(-1,1)

    for i in range(x.shape[0]):
        if y[i]==1:
            y_stream[i*60:(i+1)*60]=1
        else:
            y_stream[i*60:(i+1)*60]=0
        for h in range(60):
            mean=x[i,:,h*128:(h+1)*128].mean(axis=0)
            std=x[i,:,h*128:(h+1)*128].std(axis=0)
            s=(x[i,:,h*128:(h+1)*128]-mean)/std          
            sparse_arr= t.cat((t.zeros(128,3), c(s,0), t.zeros(128,1), c(s,16), t.zeros(128,6), c(s,1),t.zeros(128,1), c(s,17),t.zeros(128,3),
                                c(s,3), t.zeros(128,1), c(s,2), t.zeros(128,1), c(s,18),t.zeros(128,1), c(s,19), t.zeros(128,1), c(s,20),
                                t.zeros(128,1), c(s,4),t.zeros(128,1), c(s,5),t.zeros(128,1), c(s,22), t.zeros(128,1), c(s,21),t.zeros(128,1), 
                                c(s,7), t.zeros(128,1), c(s,6), t.zeros(128,1), c(s,23),t.zeros(128,1), c(s,24), t.zeros(128,1), c(s,25),
                                t.zeros(128,1), c(s,8),t.zeros(128,1), c(s,9),t.zeros(128,1), c(s,27), t.zeros(128,1), c(s,26),t.zeros(128,1),
                                c(s,11), t.zeros(128,1), c(s,10), t.zeros(128,1), c(s,15),t.zeros(128,1), c(s,28), t.zeros(128,1), c(s,29), 
                                t.zeros(128, 3), c(s,12), t.zeros(128, 1), c(s,30), t.zeros(128, 6), c(s,13), c(s,14), c(s,31), t.zeros(128,3)),dim=1).view(-1, 9, 9)

            x_stream[i*60+h]=sparse_arr  

    x_stream=x_stream.reshape(int(60*x.shape[0]),1,128,9,9)

    print(x_stream.shape, x_stream.dtype, y_stream.shape, y_stream.dtype)
  
    return x_stream, y_stream


def rbf_interpolation_scipy(arr):
  cords=np.nonzero(arr)
  y_cords=cords[0]
  x_cords=cords[1]
  d=arr[y_cords,x_cords]

  rbfi=Rbf(x_cords,y_cords,d,function='gaussian')
  b=np.linspace(0,9,16)
  xc_new,yc_new=np.meshgrid(b,b)
  return rbfi(xc_new,yc_new) 

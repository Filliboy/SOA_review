from fun import load_deap_pkl,streams_for_3dConvnet_cat,rbf_interpolation_scipy
import pickle
import torch as t


def main():
    #'/home/filip/venv_deep_eeg1/deap.pkl'
    deap=load_deap_pkl('/home/filip/venv_deep_eeg1/deap.pkl')
    x=deap['x'][:,:,128*3:8064]
    y=deap['y'][:,0]
    y=(y>=5).to(int)
    print(x.shape,y.shape)

    x_stream,y_stream=streams_for_3dConvnet_cat(x,y)

    n=x_stream.shape[0]

    x_stream_intp=t.empty((n,1,128,16,16))
    for i in range(n):
        if i%1000==0:
            print('{0} samples interpolated out of {1} ({2} % complete)'.format(i,n,i/n*100))
        for j in range(128):
            x_stream_intp[i,0,j]=t.tensor(rbf_interpolation_scipy(x_stream[i,0,j].numpy()))

    print(x_stream_intp.shape, x_stream_intp.dtype, y_stream.shape, y_stream.dtype)
    temp_dict={'x':x_stream_intp,'y':y_stream}
    #'/home/filip/venv_deep_eeg1/deap_128_16_16.pkl'
    f = open('/home/filip/venv_deep_eeg1/deap_128_16_16.pkl',"wb")
    pickle.dump(temp_dict,f)
    f.close()


if __name__ == '__main__':
    main()
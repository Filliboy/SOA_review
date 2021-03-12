import pytorch_lightning as pl
from models import R2_1d
from fun import load_deap_pkl
import torch as t

def main():
    data=load_deap_pkl('/home/filip/venv_deep_eeg1/deap_128_16_16.pkl')
    x=data['x']
    y=data['y']

    print(x.shape,x.dtype,y.shape,y.dtype)

    data_set=t.utils.data.TensorDataset(x,y)
    deap_train,deap_val=t.utils.data.random_split(data_set,[int(x.shape[0]*0.8), int(y.shape[0]*0.2)])

    train_loader=t.utils.data.DataLoader(deap_train,batch_size=16)
    val_loader=t.utils.data.DataLoader(deap_val,batch_size=16)

    model=R2_1d()

    trainer= pl.Trainer(gpus=3,progress_bar_refresh_rate=10,max_epochs=30)

    trainer.fit(model, train_loader,val_loader)

if __name__ == '__main__':
    main()

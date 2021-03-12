import pytorch_lightning as pl
import torch as t


class Residual_block(pl.LightningModule):
    def __init__(self,in_ch,out_ch,stride):
        super(Residual_block,self).__init__()

        mid=int((7*3*3*in_ch*out_ch)/(3*3*in_ch+7*out_ch))

        self.conv1=t.nn.Conv3d(in_channels=in_ch,out_channels=mid,kernel_size=(1,3,3),padding=(0,1,1),stride=(1,stride,stride))
        
        self.batchnorm1=t.nn.BatchNorm3d(mid)

        self.conv2=t.nn.Conv3d(in_channels=mid,out_channels=out_ch,kernel_size=(7,1,1),padding=(3,0,0), stride=(stride,1,1))
        
        self.batchnorm2=t.nn.BatchNorm3d(out_ch)

        self.relu=t.nn.ReLU()
        
        self.conv3=t.nn.Conv3d(in_channels=out_ch,out_channels=mid,kernel_size=(1,3,3),padding=(0,1,1))

        self.conv4=t.nn.Conv3d(in_channels=mid,out_channels=out_ch,kernel_size=(7,1,1),padding=(3,0,0))

        self.stride=stride
        if stride==2:
          self.downconv=t.nn.Conv3d(in_channels=in_ch,out_channels=out_ch, kernel_size=(1,1,1), stride=stride)
        
    def forward(self,x):
        x1=self.conv1(x)
        x1=self.batchnorm1(x1)
        x1=self.conv2(x1)
        x1=self.batchnorm2(x1)
        x1=self.relu(x1)

        x1=self.conv3(x1)
        x1=self.batchnorm1(x1)
        x1=self.conv4(x1)
        x1=self.batchnorm2(x1)
        if self.stride==2:
          x2=self.downconv(x)
        else:
          x2=x

        x=x1+x2
        x=self.relu(x)
        return x

class R2_1d(pl.LightningModule):
    def __init__(self):
        super(R2_1d,self).__init__()

        self.conv1=t.nn.Conv3d(in_channels=1,out_channels=45,kernel_size=(1,3,3),padding=(0,1,1), stride=(1,2,2))
        
        self.batchnorm1=t.nn.BatchNorm3d(45)

        self.conv2=t.nn.Conv3d(in_channels=45,out_channels=64,kernel_size=(7,1,1),padding=(3,0,0))
        
        self.batchnorm2=t.nn.BatchNorm3d(64)

        self.relu=t.nn.ReLU()

        self.res1=Residual_block(in_ch=64,out_ch=64,stride=1)

        self.res2=Residual_block(in_ch=64,out_ch=128,stride=2)

        self.res3=Residual_block(in_ch=128,out_ch=256,stride=2)

        self.res4=Residual_block(in_ch=256,out_ch=512,stride=2)

        self.avpool=t.nn.AvgPool3d(kernel_size=(16,1,1))

        self.fc1=t.nn.Linear(512,512)

        self.fc2=t.nn.Linear(512,1)

        self.sigmoid=t.nn.Sigmoid()

        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        
    def forward(self,x):
        x=self.conv1(x)
        x=self.batchnorm1(x)
        x=self.conv2(x)
        x=self.batchnorm2(x)
        x=self.relu(x)

        x=self.res1(x)
        x=self.res2(x)
        x=self.res3(x)
        x=self.res4(x)
        # print(x.shape)
        x=self.avpool(x)

        x=x.view(-1,512)
        x=self.fc1(x)
        x=self.fc2(x)
        x=self.sigmoid(x)
        return x

    def training_step(self, train_batch, batch_idx):
        x,y=train_batch
        y_hat=self.forward(x)
        loss=t.nn.functional.binary_cross_entropy(y_hat,y)
        y=y.to(int)
        acc=self.train_acc(y_hat,y)
        self.log('Train accuracy',acc)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x,y=val_batch
        y_hat=self.forward(x)
        loss=t.nn.functional.binary_cross_entropy(y_hat,y)
        y=y.to(int)
        acc=self.valid_acc(y_hat,y)
        self.log('Validation accuracy',acc)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = t.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

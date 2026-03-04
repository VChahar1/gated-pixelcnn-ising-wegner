import random
import sys
import time
import os.path
import numpy as np
from torch.optim import Adam
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch 
import torch.nn as nn
import argparse
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import py_func as pf
import gate_ising as gi
import gate_pixel as gp
import torch.distributed as dist
import pytorch_lightning as pl
from  pytorch_lightning.tuner import Tuner
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging

# with gauge fixing
#python3 3d_ising_pixel.py --beta 0.22 --batch 1500 --L 8 --epochs 1  --lr 0.003 --dev 0
class ZerosDataset(Dataset):
    def __init__(self, batch_size, var_size):
        self.batch_size = batch_size
        self.var_size = var_size

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        return torch.zeros(self.var_size,dtype=torch.float32)

    
class PixelCNN(pl.LightningModule):
    def __init__(self,m,J1,bs,lr,input,hidden):
        super().__init__()
        self.m=m
        self.J1=J1
        self.bs=bs
        self.input=input
        self.hidden=hidden
        self.loss_np =np.array([])# torch.as_tensor([]).to(self.device)
        self.magnitization=np.array([])
        self.remag=np.array([])
        self.loss=np.array([])
        self.ESS=np.array([])
        self.Wmag=np.array([])
        self.lr=lr
        self.save_hyperparameters()
        self.Net=gp.AutoregressiveColorPixelRCNN(self.input,self.input,self.hidden)
        #self.Net=gi.Gated_pixel((self.bs,1,self.m ,self.m, self.m),self.hidden,num_layers=8, k=5, padding=2)
  
    def training_step(self, batch,batch_idx):
        I_Par=torch.zeros(shape,dtype=torch.float32,device=self.device)
        Start1=time.time()
        for h in range(shape[2]):
            for w in range(shape[3]):
                for d in range(shape[4]):        
                        with torch.no_grad():
                            Net_1=(self.Net(I_Par))#forward pass of the neural network
                            rand=torch.rand((bs,1),dtype=torch.float32,device=self.device)
                            I_Par[:,:,h,w,d]=torch.where(Net_1[:,:,h,w,d] > rand[:,:],-1,1)
                        assert not I_Par.requires_grad
        End1=time.time()
        #print("Time in Tensor network",-Start1+End1)
        Net_2=self.Net(I_Par)

        Net_2=torch.where(I_Par==1,1-Net_2,Net_2)
        P_list1=torch.log(Net_2)
        P_list2=torch.sum(P_list1,dim=(1,2,3,4)) 

        loss= self.kl_loss(P_list2,I_Par)
        loss_ren=torch.mean((loss-loss.mean())*P_list2)

        self.loss_np=np.array([])
        self.loss_np=np.append(self.loss_np,loss.cpu().detach().numpy().astype(np.float128))

        self.log('loss_ren', loss_ren, on_step=False,on_epoch=True,sync_dist=True)
        self.log('loss', loss.mean()/self.J1/(m*m*n*J1), on_step=False, on_epoch=True,sync_dist=True)
        return loss_ren
        

    def kl_loss(self,Prob,I_Par): 
            with torch.no_grad():
                Energy2=pf.energy_ising3d(self.bs,m,I_Par,I_Val,self.J1).to(self.device)
                loss=Prob+Energy2

            assert not Energy2.requires_grad
            assert not loss.requires_grad           
            return loss
    

    def configure_optimizers(self):
        print("learning rate",self.lr)
        optimizer =Adam(self.parameters(), self.lr)
        scheduler={              
            "scheduler": ReduceLROnPlateau(
            optimizer=optimizer,mode="min" ,factor=0.8, patience=1, threshold=1e-5, cooldown=6000, threshold_mode="rel" ,min_lr=0.01),"monitor":"loss"}
        return [optimizer],[scheduler]
    

    def on_train_epoch_end(self) -> None:
        if ((epochs > 20) and (self.current_epoch == epochs-5)):
            self.trainer.save_checkpoint(check_dir)
        Start3=time.time()
        loss_graph=self.loss_np.mean()/self.J1/(m*m)


        weight=np.exp(-self.loss_np)
        z_hat=np.sum(weight)
#        self.remag=(self.remag*weight)/z_hat
#        self.remag=(self.remag/m*m*m*m*n).sum()
        
        ESS=(z_hat/self.bs)*(z_hat/self.bs)/(np.sum(weight*weight)/self.bs) 
        logger.experiment.add_scalar('ESS', ESS,self.current_epoch)  

#        self.Wmag=np.append(self.Wmag,self.remag)
        self.loss=np.append(self.loss,loss_graph)
        self.ESS=np.append(self.ESS,ESS)
        End3=time.time()
    def on_train_end(self) -> None:
        if ((epochs > 100) ):
            N_E=np.arange(epochs)
            np.savetxt(Mag,np.column_stack((N_E,self.loss,self.ESS)),delimiter='\t')
if __name__ == '__main__':
    Start=time.time()
    torch.set_printoptions(threshold=10000)
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta',type=float)
    parser.add_argument('--batch',type=int,default=1000)
    parser.add_argument('--epochs',type=int,default=10000)
    parser.add_argument('--L',type=int,default=4)
    parser.add_argument('--lr',type=float,default=0.05)
    parser.add_argument('--dev', nargs='+', type=int, default=1,help='number of devices')
    args=parser.parse_args()

    # for B=0.1 in 4D ising model of size 4 X 4 X 4 X 4, lr=0.05, batch = 5000
    J1=args.beta
    bs=args.batch
    m=args.L
    epochs=args.epochs
    lr=args.lr
    input=1
    hidden=18
    n=m
    print(J1,bs,m,epochs,lr)
#    dir=str(m)+"gauge_logs"+"_"+str(J1)+"_2hl"+"pl"
    dir=f'{m}_ising_logs_{J1}_gr'
#    NE="experiment"+"_"+str(lr)+"_"+str(bs)
    NE=f'experiment_{lr}_{bs}'
#   check_dir="4dcheckpoint"+"_"+str(m)+"_"+str(J1)+"_"+str(bs)+"_"+str(lr)+"_2"+"_pl"
    check_dir=f'checkpoint_{m}_{J1}_ising_3d_gr'
#    Mag="4d_Mag"+"_"+str(m)+"_"+str(J1)+"_"+str(lr)+"_"+str(bs)+".txt"
    Mag=f'Mag_{m}_{J1}_{lr}_{bs}_3d.txt'

    if len(args.dev) == 1:
        device = args.dev[0]
        dev=[device]
    elif len(args.dev) == 2:
        device1, device2 = args.dev
        dev=[device1,device2]
    # Create an instance of the ZerosDataset
   
    torch.set_printoptions(threshold=10000)
    print("Batch on each gpu",bs)
#    torch.set_printoptions(threshold=10000)
    gauge_ten=torch.as_tensor([])
    I_Val=torch.as_tensor([])
    I_Val_2=torch.as_tensor([])

    Mu=torch.zeros((bs,n,m,m))
    Mu2=torch.ones((bs,n,m,m))
    I_Val=torch.cat((I_Val,torch.nonzero(Mu2[0,:])))
    I_Val=I_Val.int()

    dyn_var=m*m*n
    print("dynamical dofs",dyn_var)
    shape=(bs,input,n,m,m)
    print(shape)
#    print("indices",I_Val)
    dataset = ZerosDataset( bs, dyn_var)

    # Create a DataLoader using the dataset
    data_loader = DataLoader(dataset, bs, shuffle=False,num_workers=1)
    if len(dev) >1:
        bs=int(bs/len(dev))

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
#    swa=pl.callbacks.StochasticWeightAveraging(swa_lrs=swa_lr,swa_epoch_start=swa_start)
    logger = pl.loggers.TensorBoardLogger(save_dir=dir, name=NE)
    

    if os.path.isfile(check_dir):
        print("starting from the chekpoint with directory",check_dir)
        model = PixelCNN.load_from_checkpoint(check_dir)
        model.lr=lr
        model.bs=bs
    else:
        print("creating checkpoint with name" ,check_dir)
        model=PixelCNN(dyn_var,J1,bs,lr,input,hidden)

    train=pl.Trainer(max_epochs=epochs,devices=dev,accelerator="gpu",gradient_clip_val=1.0,logger=logger,callbacks=[lr_monitor])
    train.fit(model,data_loader)

    Endii=time.time()

    print('Time in calculation(in mins) ',(Endii-Start)/60)

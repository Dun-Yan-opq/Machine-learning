# -*- coding: utf-8 -*-
"""
09/17/2020
簡單的 LSTM 模型建立與訓練
透過輸入一個訊號去擬合另一個訊號來感受 LSTM模型的優勢

This is a easy LSTM code!

  Wave signal A             Wave signal B
  -------------> LSTM Model ------------->
  
"""
import torch
import torch.nn as nn
from torch import optim
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import numpy as np
from matplotlib import pyplot as plt
import itertools
import time

torch.manual_seed(123)
np.random.randn(123)

# =============================================================================
# # hyper parameters
# =============================================================================
input_size = 1    # Input的特徵數量　(Input dim)
hidden_size = 36  # 隱藏單元個數     (hidden uints)
seq_len = 20      # 時序長度         (Sequential length)   
output_size = 1   # output 的維度　  (Output dim)
batch_size = 1
gap = 20
num_epochs = 2000 # 總共訓練次數     (Epoches)

h_state = None    # Hidden 狀態 (Hidden state)
c_state = None    # Cell 狀態   (Cell state)
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 檢查電腦是否有GPU (Check GPU)
# =============================================================================
# # 模型建構 (Model)
# =============================================================================
class LSTMpred(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,output_size,layer_size=1,dropout=0):
        super(LSTMpred,self).__init__()
        self.input_dim = input_size
        self.hidden_dim = hidden_size 
        self.batch_dim = batch_size
        self.output_dim = output_size 
        self.num_layers = layer_size
        self.dropout = dropout
        # 定義LSTM的架構 (Define the structure of LSTM)
        self.lstm = nn.LSTM(
                input_size=self.input_dim,   # Input的特徵數量 (Input dim)
                hidden_size=self.hidden_dim, # 隱藏單元個數    (hidden uints)
                num_layers=self.num_layers,  # LSTM的層數      (Layers of LSTM)
                bias=True,                   # 是否加入bias    (Bias or not)
                batch_first=True,            # 排列順序        (Batch first or not)
                dropout=self.dropout,        # Dropout
                bidirectional=False)         # False:單向(unidirectional), True:雙向(bidirectional)
        
        # 將 hidden的結果再轉變成我們要的輸出 (Define the output layer)
        self.out = nn.Linear(self.hidden_dim,self.output_dim) 
        self.init_weight(self.lstm) # Initialize all parameters
    def init_weight(self,m):
        # 初始化 LSTM 的參數 (Initialize all parameters in LSTM model)
        if isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    # init.orthogonal_(param.data)
                    init.xavier_normal_(param.data)
                else:
                    init.normal_(param.data)
        # 初始化 全連接層 的參數 (Initialize Fully connected layers)
        init_range=0.1 
        self.out.weight.data.uniform_(-init_range,init_range)
        
    def init_hidden(self):
        # This is what we'll initialize our hidden state
        self.batch_dim = batch_size
        return (Variable(torch.zeros(self.num_layers*1,self.batch_dim,self.hidden_dim)).to(device),  # (num_layers * num_directions, batch, hidden_size)
                Variable(torch.zeros(self.num_layers*1,self.batch_dim,self.hidden_dim)).to(device))  # (num_layers * num_directions, batch, hidden_size)
    def forward(self,x,h):
        # Forward pass through LSTM layer
        # shape of x: [batch_size, seq_len, input_size] <-------- batch_first = True
        # shape of lstm_out: [ batch_size, seq_len, num_directions *hidden_dim] <-------- batch_first = True
        # shape of self.hidden: (a, b), where a and b both 
        # have shape (num_layers, batch_size, hidden_dim).
        
        lstm_out, (h_state,c_state) = self.lstm(x,h)
        outs = [] # 保存所有的預測值 (save the values from LSTM output layer)
        for time_step in range(lstm_out.size(1)):
            outs.append(self.out(lstm_out[:,time_step,:]))

        return torch.stack(outs, dim=1), h_state, c_state
# =============================================================================
# # DataLoader
# =============================================================================
class WaveDataset(Dataset):
    def __init__(self,data,window_size,gap_size):
        x_np = data[0]
        y_np = data[1]

        self.window_size = window_size
        self.n_samples = len(x_np)
        self.gap_size = gap_size 
        self.x_data = torch.from_numpy(x_np)
        self.y_data = torch.from_numpy(y_np)
        
    def __getitem__(self, index):
        # support indexing such that dataset[i] can be used to get i-th sample
        index = index*self.gap_size
        return (self.x_data[index:index+self.gap_size], 
               self.y_data[index:index+self.gap_size])
    
    def __len__(self):
        # we can call len(dataset) to return the size
        # 最大索引值( Max Index value) = ((self.n_samples-self.window_size)/self.gap)+1
        # 切割大小(Slice number)= 最大索引值+1 (0 ~ 最大索引值)
        return int(self.n_samples/self.gap_size)
        
# =============================================================================
# # 其他參數設定與宣告 (Others)
# =============================================================================
Lstm_model = LSTMpred(input_size,hidden_size,batch_size,output_size)    # 建構LSTM模型 (Configure a LSTM model)
optimizer = optim.Adam(Lstm_model.parameters()) # Adam優化，幾乎不用調參 (Adam)
criterion = nn.MSELoss() # 因爲最終的結果是一個數值，所以損失函數用均方誤差 (MSE)

# GPU
Lstm_model.to(device)

Lstm_model.train()
training_loss=[]
plt.figure(1)

start, end = 0 * np.pi, (0+2)*np.pi # 一個時間週期 (A time period)
num_data = 120    # 資料點數(Time samples)
steps = np.linspace(start, end, num_data, dtype=np.float32)
show_steps = np.linspace(0, 2*np.pi, num_data, dtype=np.float32)

# 產生波形 x_np(Input) y_np(Target) (Generate Wave Signal)
x_np = np.cos(steps) # Input      <-------------- 可任意更改 (Changable)
y_np = np.sin(steps)*np.cos(4*steps) # Target <-------------- 可任意更改 (Changable)

# Slice all data
All_data=(x_np,y_np)
dataset=WaveDataset(All_data,seq_len,gap)
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          drop_last=False,
                          num_workers=0)
# =============================================================================
# # Main
# =============================================================================                      
if __name__ == '__main__':
    start_t = time.time() # Record the time 
    total_loss=[]
    for epoch in range(0,num_epochs,2):
        hh=Lstm_model.init_hidden() # 初始化(All set zeros)
        show=[]
        target_append=[]
        prediction_append=[]
        target_buffer=torch.tensor([]).to(device)
        predict_buffer=torch.tensor([]).to(device)
        loss_sum=0
        for count,(x,y) in enumerate(train_loader):
            x = Variable(x.reshape(batch_size,seq_len,input_size)).to(device) # shape (batch, time_step, input_size)
            y = Variable(y.reshape(batch_size,seq_len,input_size)).to(device)
            Lstm_model.zero_grad() # 清除LSTM的上個數據的偏微分暫存值，否則會一直累加
            
            prediction, h_state,c_state = Lstm_model(x,hh) # Lstm output
            hh=(Variable(h_state.data.detach()).to(device),Variable(c_state.data.detach()).to(device))
            
            Target=y.view(-1)
            prediction=prediction.view(-1)

            target_append.append(Target)
            prediction_append.append(prediction)

            if (count+1)%5==0 or count == 5:
                for i in range(len(target_append)):
                   target_buffer=torch.cat((target_buffer,target_append[i]))
                   predict_buffer=torch.cat((predict_buffer,prediction_append[i]))
                   
                loss=criterion(target_buffer,predict_buffer)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum+=loss

                predict_buffer=predict_buffer.cpu()
                show.append(predict_buffer.data.numpy().flatten())
                
                target_buffer=torch.tensor([]).to(device)  # Empty target_buffer
                predict_buffer=torch.tensor([]).to(device) # Empty predict_buffer
                target_append=[]      # Empty target_append
                prediction_append=[]  # Empty prediction_append
                
                #每訓練20個批次可視化一下效果，並打印一下loss (Show the plot and print loss value per 20)
                if (epoch)%20==0 and count == int(num_data/(seq_len*batch_size))-1: 
                    show_out=[]
                    print("EPOCHS: {},Loss:{:4f}".format(epoch,loss))
                    plt.cla() # 清除axes，即當前 figure 中的活動的axes，但其他axes保持不變。
                    show_out = list(itertools.chain.from_iterable(show))
                    # Plot Input, Target, Predict data
                    plt.plot(show_steps, x_np.flatten(), 'g-', label="Input")
                    plt.plot(show_steps, y_np.flatten(), 'r-', label="Target")
                    plt.plot(show_steps,show_out,'b-',label="Predict")
                    plt.legend()
                    
                    plt.pause(0.01)
        total_loss.append(loss_sum)   
# 繪製訓練總誤差 (Plot training loss)        
plt.figure(2)
plt.plot(total_loss, label="Training loss")
plt.legend()
plt.show()
all_time=time.time() - start_t

print('Running time:{}min'.format(all_time/60))     
print("Finish!!!")          
    


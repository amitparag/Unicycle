import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import Datagen
from value_network import ValueNet
from residual_network import ResidualNet



def train(size_of_training_dataset = 3000,
          xrange_training_data     = [-2.1, 2.1],
          yrange_training_data     = [-2.1, 2.1],
          zrange_training_data     = [-2.*np.pi, 2.*np.pi], #theta
          # DDP solver params 
          horizon                  = 30,
          stop                     = 1e-9,
          maxiters                 = 1000,
          state_weight             = 1.,
          control_weight           = 1.,
          # Neural Network params
          input_features           = 3,
          output_features          = 1,
          nhiddenunits             = 64,
          activation               = nn.Tanh(),
          learning_rate            = 1e-3,
          epochs                   = 10000,
          batchsize                = 1000,
          name                     = 'value',  
          save_name                = None):
    """
    
    Initialize and train a new feedforward value network.
    
    @params:
        # Parameters of training dataset.
        
        1: size_of_training_dataset.
        2: xrange_training_data = the range of x to sample from, when creating the training data.
        3: yrange_training_data = the range of y to sample from, when creating the training data.
        4: zrange_training_data = the range of z to sample from, when creating the training data.
            Default ranges of x, y, theta:
                x -> [-2.1, 2.1]
                y -> [-2.1, 2.1]
                z -> [-2pi, 2pi] = theta
        
        # Parameters given to crocoddyl to generate training data
        
        5: horizon        = time horizon for the ddp solver, T
        6: stop           = ddp.th_stop
        7: maxiters       = maximum iterations allowed for solver
        8: state_weight   = weight of the state vector
        9: control_weight = weight of the control vector
        
        # Parameters of the neural network
        
        10: input_features  = number of columns of the dataset. 3, for x, y, z
        11: output_features = number of columns to ytrain. 1, since we are modelling ddp.cost
        12: nhiddenunits    = number of units in each hidden layer
        13: activation      = use either tanh() or relu()
        14: learning_rate   = 1e-3
        15: epochs          = number of epochs for training
        16: batchsize       = batchsize of data during training
        17: save_name       = if a str is given, then the net will be saved. 
        
    """

    ##.......................... Training
    
    # Sample random positions for xtrain
    positions = Datagen.random_positions(size = size_of_training_dataset,
                                         xlim = xrange_training_data,
                                         ylim = yrange_training_data,
                                         zlim = zrange_training_data,
                                         as_tensor=True)
    
    # Corresponding ddp.cost for ytrain    
    values    = Datagen.values(positions=positions,
                               horizon= horizon,
                               stop= stop,
                               maxiters=maxiters,
                               state_weight=state_weight,
                               control_weight=control_weight,
                               as_tensor=True)
    
    

    # Torch dataloader
    dataset = torch.utils.data.TensorDataset(positions,values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batchsize) 

    # Initialize an untrained  network
    
    if name.lower=="value":
        
        net = ValueNet(in_features = input_features,
                       out_features = output_features,
                       nhiddenunits = nhiddenunits,
                       activation   = activation)
    else:
        net = ResidualNet(in_features = input_features,
                          out_features = output_features,
                          nhiddenunits = nhiddenunits,
                          activation   = activation)
        

    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)  

    net.train()
    print("\n Training ... \n")
    for epoch in tqdm(range(epochs)):        
        for data, target in dataloader: 

            outputs = net(data)
            loss = criterion(outputs, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    ##........................... Validation
    
    # Validation dataset
    xtest     = Datagen.random_positions(size=1000, as_tensor=True)
    ytest     = Datagen.values(xtest)
    net.eval()
    ypred = net(xtest)
    error = ypred.detach() - ytest
    print(f'Mean Error:{torch.mean(error)}')
    
    if save_name is not None:
        torch.save(net, "../networks/"+save_name+".pth")
        
    else: return net
        

if __name__=='__main__':
    
    import torch
    
    train(name="value", save_name='value')
    train(name='residual',nhiddenunits=128,output_features=5,save_name='residual')
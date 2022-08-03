import torch.nn as nn
import torch


c1 = 12 #b/c single time-series
c2 = 4 #4
c3 = 16 #16
c4 = 32 #32
k=7 #kernel size #7 
s=3 #stride #3
#num_classes = 3

class ConvNet(nn.Module):
    
    """ CNN for Self-Supervision """
    
    def __init__(self,dropout_type,p1,p2,p3,nencoders=1,embedding_dim=256,trial='',device=''):
        super(ConvNet,self).__init__()
        
        self.embedding_dim = embedding_dim
        
        if dropout_type == 'drop1d':
            self.dropout1 = nn.Dropout(p=p1) #0.2 drops pixels following a Bernoulli
            self.dropout2 = nn.Dropout(p=p2) #0.2
            self.dropout3 = nn.Dropout(p=p3)
        elif dropout_type == 'drop2d':
            self.dropout1 = nn.Dropout2d(p=p1) #drops channels following a Bernoulli
            self.dropout2 = nn.Dropout2d(p=p2)
            self.dropout3 = nn.Dropout2d(p=p3)
        
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.maxpool = nn.MaxPool1d(2)
        self.trial = trial
        self.device = device
        
        self.view_modules = nn.ModuleList()
        self.view_linear_modules = nn.ModuleList()
        for n in range(nencoders):
            self.view_modules.append(nn.Sequential(
            nn.Conv1d(c1,c2,k,s),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout1,
            nn.Conv1d(c2,c3,k,s),
            nn.BatchNorm1d(c3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout2,
            nn.Conv1d(c3,c4,k,s),
            nn.BatchNorm1d(c4),
            nn.ReLU(),
            nn.MaxPool1d(2),
            self.dropout3
            ))
            self.view_linear_modules.append(nn.Linear(c4*10,self.embedding_dim))
                        
    def forward(self,x):
        """ Forward Pass on Batch of Inputs 
        Args:
            x (torch.Tensor): inputs with N views (BxSxN)
        Outputs:
            h (torch.Tensor): latent embedding for each of the N views (BxHxN)
        """
        batch_size = x.shape[0]
        nsamples = x.shape[2]
        nviews = x.shape[3]
        # print("x shape {}".format(x.shape))
        # print("nsamples {}".format(nsamples))
        # print("nviews {}".format(nviews))
        # nviews = x.shape[2]
        latent_embeddings = torch.empty(batch_size,self.embedding_dim,nviews,device=self.device)
        for n in range(nviews):       
            """ Obtain Inputs From Each View """
            h = x[:,:,:,n]
            # h = x[:,:,n]
            
            if self.trial == 'CMC':
                h = self.view_modules[n](h) #nencoders = nviews
                h = torch.reshape(h,(h.shape[0],h.shape[1]*h.shape[2]))
                h = self.view_linear_modules[n](h)
            else:
                h = self.view_modules[0](h) #nencoder = 1 (used for all views)
                h = torch.reshape(h,(h.shape[0],h.shape[1]*h.shape[2]))
                h = self.view_linear_modules[0](h)

            latent_embeddings[:,:,n] = h
        
        return latent_embeddings

class second_cnn_network(nn.Module):
    
    def __init__(self,first_model,noutputs,embedding_dim=256):
        super(second_cnn_network,self).__init__()
        self.first_model = first_model
        self.hidden = nn.Linear(embedding_dim,embedding_dim)
        self.linear = nn.Linear(embedding_dim,noutputs)
        self.noutputs = noutputs

    def forward(self,x):
        h = self.first_model(x)
        h = h.squeeze() #to get rid of final dimension from torch.empty before
        h = self.hidden(h)
        h = torch.nn.functional.relu(h)
        h = self.linear(h)
        output = torch.nn.functional.softmax(h, dim=1)
        return output


class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation = dilation, padding = 3, bias=True)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out       

class se_block(nn.Module):
    def __init__(self,in_layer, out_layer):
        super(se_block, self).__init__()
        
        self.conv1 = nn.Conv1d(in_layer, out_layer//8, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(out_layer//8, in_layer, kernel_size=1, padding=0)
        self.fc = nn.Linear(1,out_layer//8)
        self.fc2 = nn.Linear(out_layer//8,out_layer)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):

        x_se = nn.functional.adaptive_avg_pool1d(x,1)
        x_se = self.conv1(x_se)
        x_se = self.relu(x_se)
        x_se = self.conv2(x_se)
        x_se = self.sigmoid(x_se)
        
        x_out = torch.add(x, x_se)
        return x_out

class re_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, dilation):
        super(re_block, self).__init__()
        
        self.cbr1 = conbr_block(in_layer,out_layer, kernel_size, 1, dilation)
        self.cbr2 = conbr_block(out_layer,out_layer, kernel_size, 1, dilation)
        self.seblock = se_block(out_layer, out_layer)
    
    def forward(self,x):
        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.seblock(x_re)
        x_out = torch.add(x, x_re)
        return x_out   
        

class unet_1D(nn.Module):
    def __init__(self, input_dim, embedding_dim, kernel_size, depth, nmasks, device="cpu"):
        super(unet_1D, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.kernel_size = kernel_size
        self.depth = depth
        self.nmasks = nmasks
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5, padding=4)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)
        
        self.layer1 = self.down_layer(self.input_dim, self.embedding_dim, self.kernel_size, 1, self.depth)
        self.layer2 = self.down_layer(self.embedding_dim, int(self.embedding_dim*2), self.kernel_size, 5, self.depth)
        self.layer3 = self.down_layer(int(self.embedding_dim*2)+int(self.input_dim), int(self.embedding_dim*3), self.kernel_size, 5, self.depth)
#         self.layer4 = self.down_layer(int(self.embedding_dim*3)+int(self.input_dim), int(self.embedding_dim*4), self.kernel_size, 5, self.depth)
#         self.layer5 = self.down_layer(int(self.embedding_dim*4)+int(self.input_dim), int(self.embedding_dim*5), self.kernel_size, 4, self.depth)

#         self.cbr_up1 = conbr_block(int(self.embedding_dim*7), int(self.embedding_dim*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.embedding_dim*5), int(self.embedding_dim*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.embedding_dim*3), self.embedding_dim, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')
        
        self.outcov = nn.Conv1d(self.embedding_dim, self.nmasks, kernel_size=self.kernel_size, stride=1,padding = 3)
    
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        pool_x1 = self.AvgPool1D1(x)
        # pool_x2 = self.AvgPool1D2(x)
        # pool_x3 = self.AvgPool1D3(x)
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        x = self.layer3(x)

        up = self.upsample(x)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        # out = torch.sigmoid(out)
        if self.nmasks == 1:
            out = nn.functional.sigmoid(out)
        else:
            out = nn.functional.softmax(out, dim=1) # dim=1 is across the 12 leads 
        
        return out
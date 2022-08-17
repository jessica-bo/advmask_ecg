import torch.nn as nn
import torch

# count_param(unet1d)=247,077,697 with depth=1, kernel_size=5
class unet1D(nn.Module):
    def __init__(self, input_dim, embedding_dim, depth, nmasks):
        super(unet1D, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.kernel_size = 5
        self.depth = depth
        self.nmasks = nmasks
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5, padding=4)
        self.AvgPool1D2 = nn.AvgPool1d(input_dim, stride=25)
        self.AvgPool1D3 = nn.AvgPool1d(input_dim, stride=125)
        
        self.layer1 = self.down_layer(self.input_dim, self.embedding_dim, self.kernel_size,1, self.depth)
        self.layer2 = self.down_layer(self.embedding_dim, int(self.embedding_dim*2), self.kernel_size,5, self.depth)
        self.layer3 = self.down_layer(int(self.embedding_dim*2)+int(self.input_dim), int(self.embedding_dim*3), self.kernel_size,5, self.depth)
        self.layer4 = self.down_layer(int(self.embedding_dim*3)+int(self.input_dim), int(self.embedding_dim*4), self.kernel_size,5, self.depth)
        self.layer5 = self.down_layer(int(self.embedding_dim*4)+int(self.input_dim), int(self.embedding_dim*5), self.kernel_size,4, self.depth)

        self.cbr_up1 = conbr_block(int(self.embedding_dim*7), int(self.embedding_dim*3), self.kernel_size, 1, 1)
        self.cbr_up2 = conbr_block(int(self.embedding_dim*5), int(self.embedding_dim*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.embedding_dim*3), self.embedding_dim, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')
        
        self.outcov = nn.Conv1d(self.embedding_dim, self.nmasks, kernel_size=self.kernel_size, stride=1,padding=2)
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        
        pool_x1 = self.AvgPool1D1(x)
        pool_x2 = self.AvgPool1D2(x)
        pool_x3 = self.AvgPool1D3(x)
        
        #############Encoder#####################
        
        out_0 = self.layer1(x)
        out_1 = self.layer2(out_0)
        
        x = torch.cat([out_1,pool_x1],1)
        out_2 = self.layer3(x)
        
        x = torch.cat([out_2,pool_x2],1)
        x = self.layer4(x)
        
        #############Decoder####################
        
        up = self.upsample1(x)
        up = torch.cat([up,out_2],1)
        up = self.cbr_up1(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_1],1)
        up = self.cbr_up2(up)
        
        up = self.upsample(up)
        up = torch.cat([up,out_0],1)
        up = self.cbr_up3(up)
        
        out = self.outcov(up)
        # out = torch.sigmoid(out)
        if self.nmasks == 1:
            out = torch.sigmoid(out)
        else:
            out = nn.functional.softmax(out, dim=1) # dim=1 is across the 12 leads 
        
        return out

# count_param(unet1dsmall)=65,766,529 with depth=1, kernel_size=5
class unet1Dsmall(nn.Module):
    def __init__(self, input_dim, embedding_dim, depth, nmasks):
        super(unet1Dsmall, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.kernel_size = 5
        self.depth = depth
        self.nmasks = nmasks
        
        self.AvgPool1D1 = nn.AvgPool1d(input_dim, stride=5, padding=4)
        
        self.layer1 = self.down_layer(self.input_dim, self.embedding_dim, self.kernel_size, 1, self.depth)
        self.layer2 = self.down_layer(self.embedding_dim, int(self.embedding_dim*2), self.kernel_size, 5, self.depth)
        self.layer3 = self.down_layer(int(self.embedding_dim*2)+int(self.input_dim), int(self.embedding_dim*3), self.kernel_size, 5, self.depth)
        
        self.cbr_up2 = conbr_block(int(self.embedding_dim*5), int(self.embedding_dim*2), self.kernel_size, 1, 1)
        self.cbr_up3 = conbr_block(int(self.embedding_dim*3), self.embedding_dim, self.kernel_size, 1, 1)
        self.upsample = nn.Upsample(scale_factor=5, mode='nearest')
        self.upsample1 = nn.Upsample(scale_factor=5, mode='nearest')
        
        self.outcov = nn.Conv1d(self.embedding_dim, self.nmasks, kernel_size=self.kernel_size, stride=1, padding=2)
        
    def down_layer(self, input_layer, out_layer, kernel, stride, depth):
        block = []
        block.append(conbr_block(input_layer, out_layer, kernel, stride, 1))
        for i in range(depth):
            block.append(re_block(out_layer,out_layer,kernel,1))
        return nn.Sequential(*block)
            
    def forward(self, x):
        pool_x1 = self.AvgPool1D1(x)
        
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
            out = torch.sigmoid(out)
        else:
            out = nn.functional.softmax(out, dim=1) # dim=1 is across the 12 leads 
        
        return out


class conbr_block(nn.Module):
    def __init__(self, in_layer, out_layer, kernel_size, stride, dilation):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=2, bias=True)
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


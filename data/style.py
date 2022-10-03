"""
Adapted from @jone1222
Source: https://github.com/jone1222/DG-Feature-Stylization/blob/e3d8380f8714324fbe0b387bcc8379af02eec2fb/dassl/modeling/backbone/resnet_stylization.py
"""

import torch 

def encode_LL_HH(x, interpolate_mode='nearest', kernel=10):
    pooled = torch.nn.functional.avg_pool1d(x, kernel)
    up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2)], mode=interpolate_mode)
    HH = x - up_pooled
    LL = up_pooled
    return LL, HH

def stylize_feature(LL, scaling_factor=1):
    scaling_factor = scaling_factor
    B, C, W = LL.shape
    LL_cp = LL.view(C, -1)  # MEAN / STD : (C) -> batch-wise mean of each-channel

    # Calculate batch-wise statistics
    mean_LL = torch.mean(LL_cp, dim=1) # batch-wise mean
    std_LL = torch.std(LL_cp, dim=1) # batch-wise std

    # Calculate channel-wise statistics of mean and std vector
    # mu_hat_LL = torch.tensor(0.4462575614452362).cuda() 
    # sigma_hat_LL = torch.tensor(0.005226934794336557).cuda() 
    # mu_tilde_LL = torch.tensor(0.13094377517700195).cuda()  
    # sigma_tilde_LL = torch.tensor(0.004083267413079739).cuda()  
    mu_hat_LL = mean_LL.mean()
    sigma_hat_LL = mean_LL.std()
    mu_tilde_LL = std_LL.mean()
    sigma_tilde_LL = std_LL.std()

    # Sample new style vectors from the manipulated distribution
    mu_new = torch.normal(mu_hat_LL.view(1, 1).repeat(B, C), scaling_factor * sigma_hat_LL.view(1, 1).repeat(B, C)) #output : (B, C)
    sigma_new = torch.normal(mu_tilde_LL.view(1, 1).repeat(B, C), scaling_factor * sigma_tilde_LL.view(1,1).repeat(B, C))

    mu_new_reshape = mu_new.view(B, C, 1).repeat(1, 1, W)
    sigma_new_reshpae = sigma_new.view(B, C, 1).repeat(1, 1, W)

    # Equation 6 ~ Normalize original feature with batch statistics
    normalized_LL = (LL - mean_LL.view(1, C, 1).repeat(B, 1, W)) / std_LL.view(1, C, 1).repeat(B, 1, W)
    # Equation 6 ~ Affine transformation with sampled style vectors
    stylized_LL = sigma_new_reshpae * normalized_LL + mu_new_reshape

    return stylized_LL
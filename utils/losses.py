"""
Adapted from @danikiyasseh
Source: https://github.com/danikiyasseh/CLOCS/blob/master/prepare_miscellaneous.py
"""

import torch

def simclr_loss_fn(latent_embeddings, positive_pairing="SimCLR", temperature=0.1):
    """ Calculate NCE Loss For Latent Embeddings in Batch 
    Args:
        latent_embeddings (torch.Tensor): embeddings from model for different perturbations of same instance (BxHxN)
    Outputs:
        loss (torch.Tensor): scalar NCE loss 
    """
    loss = 0

    view1_array = latent_embeddings[0] #(BxH)
    view2_array = latent_embeddings[1] #(BxH)

    norm1_vector = view1_array.norm(dim=1).unsqueeze(0)
    norm2_vector = view2_array.norm(dim=1).unsqueeze(0)

    sim_matrix = torch.mm(view1_array,view2_array.transpose(0,1))
    norm_matrix = torch.mm(norm1_vector.transpose(0,1),norm2_vector)
    argument = sim_matrix/(norm_matrix*temperature)
    sim_matrix_exp = torch.exp(argument)
    
    if positive_pairing in ['SimCLR', 'CMSC']:
        self_sim_matrix1 = torch.mm(view1_array,view1_array.transpose(0,1))
        self_norm_matrix1 = torch.mm(norm1_vector.transpose(0,1),norm1_vector)
        argument = self_sim_matrix1/(self_norm_matrix1*temperature)
        self_sim_matrix_exp1 = torch.exp(argument)
        self_sim_matrix_off_diagonals1 = torch.triu(self_sim_matrix_exp1,1) + torch.tril(self_sim_matrix_exp1,-1)
        
        self_sim_matrix2 = torch.mm(view2_array,view2_array.transpose(0,1))
        self_norm_matrix2 = torch.mm(norm2_vector.transpose(0,1),norm2_vector)
        argument = self_sim_matrix2/(self_norm_matrix2*temperature)
        self_sim_matrix_exp2 = torch.exp(argument)
        self_sim_matrix_off_diagonals2 = torch.triu(self_sim_matrix_exp2,1) + torch.tril(self_sim_matrix_exp2,-1)

        denominator_loss1 = torch.sum(sim_matrix_exp,1) + torch.sum(self_sim_matrix_off_diagonals1,1)
        denominator_loss2 = torch.sum(sim_matrix_exp,0) + torch.sum(self_sim_matrix_off_diagonals2,0)
        
        diagonals = torch.diag(sim_matrix_exp)
        loss_term1 = -torch.mean(torch.log(diagonals/denominator_loss1))
        loss_term2 = -torch.mean(torch.log(diagonals/denominator_loss2))
        loss += loss_term1 + loss_term2
        loss_terms = 2

    # elif positive_pairing in ['CMSC','CMLC','CMSMLC']: #ours #CMSMLC = positive examples are same instance and same patient
    #     # triu_elements = sim_matrix_exp[rows1,cols1]
    #     # tril_elements = sim_matrix_exp[rows2,cols2]
    #     diag_elements = torch.diag(sim_matrix_exp)
        
    #     triu_sum = torch.sum(sim_matrix_exp,1)
    #     tril_sum = torch.sum(sim_matrix_exp,0)
        
    #     loss_diag1 = -torch.mean(torch.log(diag_elements/triu_sum))
    #     loss_diag2 = -torch.mean(torch.log(diag_elements/tril_sum))
        
    #     # loss_triu = -torch.mean(torch.log(triu_elements/triu_sum[rows1]))
    #     # loss_tril = -torch.mean(torch.log(tril_elements/tril_sum[cols2]))
        
    #     loss = loss_diag1 + loss_diag2
    #     loss_terms = 2

    #     # if len(rows1) > 0:
    #     #     loss += loss_triu #technically need to add 1 more term for symmetry
    #     #     loss_terms += 1
        
    #     # if len(rows2) > 0:
    #     #     loss += loss_tril #technically need to add 1 more term for symmetry
    #     #     loss_terms += 1
        
    loss = loss/(loss_terms)
    return loss
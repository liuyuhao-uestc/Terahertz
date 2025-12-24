import torch
import torch.nn as nn


class Interleaver(nn.Module):
    def __init__(self,packet_size = 4):
        super(Interleaver, self).__init__()
        self.packet_size = packet_size
        self.indices = None
        self.inverse_indices = None
        self.latent_sizes = None

    ### 交织：
    ### (b,w1,h1),(b,w2,h2),(b,w3,h3)
    ### (b,l1),(b,l2),(b,l3)
    ### (b,N = l1+l2,l3)
    ### (b,打乱)
    def interleave(self, batch_list):
        batch_sizes = [b.shape[0] for b in batch_list]
        assert batch_sizes[0]==batch_sizes[1] and batch_sizes[1] == batch_sizes[2]
        B = batch_sizes[0]
        batch_1 = batch_list[0].view(B,-1)
        batch_2 = batch_list[1].view(B,-1)
        batch_3 = batch_list[2].view(B,-1)
        self.latent_sizes = [b.shape[1] for b in [batch_1, batch_2, batch_3]]
        batch = torch.cat((batch_1,batch_2,batch_3),dim=1) #[B,-1]
        N  = batch.shape[1]
        self.indices = torch.randperm(N)
        self.inverse_indices = torch.zeros_like(self.indices)
        self.inverse_indices[self.indices] = torch.arange(N)
        shuffled_batch = torch.zeros_like(batch)
        for b in range(B):
            shuffled_batch[b] = batch[b][self.indices]
        shuffled_batch = shuffled_batch.view(B, -1, self.packet_size)
        return shuffled_batch

    ### 反交织
    ### (b,打乱)
    ### (b,N = l1+l2,l3)
    def deinterleave(self, batch):
        B, num_packet, packet_size = batch.shape
        flatten = batch.view(B, -1)
        deinterleave_batch = torch.zeros_like(flatten)
        for b in range(B):
            deinterleave_batch[b] = flatten[b][self.inverse_indices]
        #deinterleave_batch = deinterleave_batch.reshape(B, num_packet, packet_size)
        len1 = self.latent_sizes[0]
        #print("x",deinterleave_batch.shape)
        batch_1 = deinterleave_batch[:, :len1].view(B,3,64*64).permute(0,2,1)
        len2 = self.latent_sizes[1]
        batch_2 = deinterleave_batch[:,len1:len1+len2].view(B,3,64*64).permute(0,2,1)
        len3 = self.latent_sizes[2]
        batch_3 = deinterleave_batch[:,len1+len2:].view(B,3,64*64).permute(0,2,1)
        return batch_1, batch_2, batch_3
'''
if __name__ == '__main__':
    x = torch.arange(24).reshape(2, 2, 6)  # (2,12)
    y = torch.arange(36).reshape(2, 3, 6)
    z = torch.arange(12).reshape(2, 1, 6)
    interleaver = Interleaver()
    q = interleaver.interleave([x,y,z])
    b1,b2,b3 = interleaver.deinterleave(q)
'''
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, d_word_embedding, d_h):
        # TODO: add embedding layer]
        self.right_RNN = myRNN(d_word_embedding, d_h)
        self.left_RNN = myRNN(d_word_embedding, d_h)


    def forward(self, input, h_0):
        # TODO: Initialize h_0 directly
        h_prev_1 = h_0
        h_prev_2 = C_0
        # TODO: fix the h_left dimension with input dimension
        #       Including: batch_size, max_sent_len
        #                  Also, fix the shape so that
        #                  the model forward the tensor correctly
        h_left = torch.zeros(input.shape[1], d_h)
        h_right = torch.zeros(input.shape[1], d_h)
        # TODO: fix the memory issue
        counter = 0
        for i in input:
            h_prev_1= self.right_RNN(i, h_prev_1)
            h_left[counter] = h_prev
            counter += 1
        counter = 0
        for i in torch.flip(input, [1]):
            h_prev_2 = self.left_RNN(i, h_prev_2)
            h_right[counter] = h_prev_2
            counter += 1
        return h_left, h_right



class myRNN(nn.Module):
    def __init__(self, d_word_embedding, d_h):
        super(BiRNN, self).__init__()

        self.z_i_x = nn.Linear(d_word_embedding, d_h)
        self.z_i_h = nn.Linear(d_h, d_h)
        self.r_i_x = nn.Linear(d_word_embedding, d_h)
        self.z_i_h = nn.Linear(d_h, d_h)
        self.h_i_x = nn.Linear(d_word_embedding, d_h)
        self.h_i_h = nn.Linear(d_h, d_h)

    def forward(x, h_prev):
        # This is from GRU, similar functionality as LSTM, which 
        # reduce the possibility for the explosion/vanishment of gradient
        z_i = nn.Sigmoid()(self.z_i_x(x)+self.z_i_h(h_prev))
        r_i = nn.Sigmoid()(self.r_i_x(x)+self.r_i_h(h_prev))
        h_i_ = nn.Tanh()(self._h_i_x(x)+self.h_i_h(r_i * h_prev))
        # TODO: Fix this
        h_i = (1-z_i) * h_prev + z_i * h_i_
        return h_i



class Decoder(nn.Module):
    def __init__(self, d_h, d_s, d_word_embedding):
        super(Decoder, self).__init__()

        self.attention_layer = AttentionLayer():
        # TODO: Finish the code

    def foward(x):
        pass

class AttentionLayer(nn.Module):
    def __init__(self, d_word_embedding, 
        d_h, d_s, max_sent_len, d_v):
        super(AttentionLayer, self).__init__()

        self.attention_s = nn.Linear()
        self.attention_h = nn.Linear()
        self.attention = nn.Linear(d_v, d_word_embedding)

    def foward(s_prev, h):
        # TODO: expand s_prev to n*d_s
        # TODO: Fix the dim_h, h: n*2d_h
        # TODO: Fix this broken outline code
        e_i = nn.Tanh()(self.attention_s(s_prev)+
                        self.attention_h(h))
        attention = nn.Softmax()(self.attention(e_i))
        return attention


# TODO: Implement the Decoder with following mechanism:
##      1. Attention Layer
##      2. RNN for s

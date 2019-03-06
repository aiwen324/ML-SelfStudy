import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
import Constants


class Encoder(nn.Module):
    def __init__(self, d_word_embedding, d_h, src_vocab_size):
        # TODO: add embedding layer]
        super().__init__()
        self.word_embed_layer = nn.Embedding(src_vocab_size, 
            d_word_embedding, padding_idx=Constants.PAD)
        self.right_RNN = myRNN(d_word_embedding=d_word_embedding, d_h=d_h)
        self.left_RNN = myRNN(d_word_embedding=d_word_embedding, d_h=d_h)
        self.d_h = d_h


    def forward(self, x):
        # Initialize h_0 for both direction
        h_prev_1 = torch.zeros(x.shape[0], self.d_h, device='cuda')
        h_prev_2 = torch.zeros(x.shape[0], self.d_h, device='cuda')
        # print('## debug msg: input:', x.shape)
        embedded_x = self.word_embed_layer(x)
        # print('## debug msg: embedded_x:', embedded_x.shape)
        # We need to change reshape the tensor for training purpose
        # (batch_size x max_sequence_len x d_embed) --> (max_sequence_len x batch_size x d_embed)
        embedded_x = embedded_x.permute(1, 0, 2)
        # h_left shape: batch_size x max_sequence_len x d_h
        h_left = torch.zeros(x.shape[0], embedded_x.shape[0], self.d_h, device='cuda')
        # print("## debug msg: h_left shape is", h_left.shape)
        h_right = torch.zeros(x.shape[0], embedded_x.shape[0], self.d_h, device='cuda')
        # TODO: fix the memory issue
        counter = 0
        for i in embedded_x:
            h_prev_1= self.right_RNN(i, h_prev_1)
            h_left[:,counter,:] = h_prev_1
            counter += 1
        counter = 0
        for i in torch.flip(embedded_x, [1]):
            h_prev_2 = self.left_RNN(i, h_prev_2)
            h_right[:,counter,:] = h_prev_2
            counter += 1
        # print(h_left.shape, h_right.shape)
        output = torch.cat((h_left, h_right), dim=2)
        # print("## debug msg: output shape is", output.shape)
        return output



class myRNN(nn.Module):
    def __init__(self, d_word_embedding, d_h):
        super(myRNN, self).__init__()

        self.z_i_x = nn.Linear(d_word_embedding, d_h)
        self.z_i_h = nn.Linear(d_h, d_h)
        self.r_i_x = nn.Linear(d_word_embedding, d_h)
        self.r_i_h = nn.Linear(d_h, d_h)
        self.h_i_x = nn.Linear(d_word_embedding, d_h)
        self.h_i_h = nn.Linear(d_h, d_h)

    def forward(self, x, h_prev):
        # This is from GRU, similar functionality as LSTM, which 
        # reduce the possibility for the explosion/vanish of gradient
        # print(x.shape, h_prev.shape)
        z_i = nn.Sigmoid()(self.z_i_x(x)+self.z_i_h(h_prev))
        r_i = nn.Sigmoid()(self.r_i_x(x)+self.r_i_h(h_prev))
        h_i_ = nn.Tanh()(self.h_i_x(x)+self.h_i_h(r_i * h_prev))
        # TODO: Fix this
        h_i = (1-z_i) * h_prev + z_i * h_i_
        return h_i



class Decoder(nn.Module):
    def __init__(self, d_h, d_s, d_word_embedding, tgt_vocab_size, d_l):
        super(Decoder, self).__init__()

        self.attention_layer = AttentionLayer(d_word_embedding=d_word_embedding, d_h=d_h, d_s=d_s,
                                              d_v=d_s)
        self.decoder_rnn = DeRNN(d_h=d_h, d_s=d_s, 
                                 d_word_embedding=d_word_embedding,
                                 tgt_vocab_size=tgt_vocab_size,
                                 d_l=d_l)
        self.d_s = d_s
        self.tgt_vocab_size=tgt_vocab_size
    def forward(self, hidden_matrix, src_max_sent_len, tgt_max_sent_len):
        # print(len(hidden_matrix))
        s_prev = torch.zeros(hidden_matrix.shape[0], 1, self.d_s, device='cuda')
        y_prev = torch.zeros(hidden_matrix.shape[0], 1, dtype=torch.long, device='cuda')
        output = torch.zeros(hidden_matrix.shape[0], tgt_max_sent_len, self.tgt_vocab_size, device='cuda')
        for i in range(tgt_max_sent_len):
            attention = self.attention_layer(s_prev, hidden_matrix, src_max_sent_len)
            # print('## debug msg: attention before view', attention.shape)
            # print(attention)
            # print(attention.shape, hidden_matrix.shape)
            attention = attention.view(-1, 1, src_max_sent_len)
            # print('## debug msg: attention after view', attention.shape)
            # print(attention)
            # print('## debug msg: hidden_matrix ', hidden_matrix.shape)
            # print(hidden_matrix)
            # attention shape: batch_size x 1 x src_max_sent_len
            # hidden_matrix shape: batch_size x src_max_sent_len x 2d_s
            c_i = torch.bmm(attention, hidden_matrix)
            # print(c_i.shape)
            # print(s_prev.shape)
            # output_i shape: batch_size x 1 x tgt_vocab_size
            output_i, s_prev = self.decoder_rnn(s_prev, y_prev, c_i)
            # print('## debug msg: output_i is: ', output_i.shape)
            # print(output_i)
            y_prev = output_i.max(2)[1]
            # print('## debug msg: y_prev is:', y_prev.shape)
            # print(y_prev)
            # print(y_prev.shape)
            # print(s_prev.shape)
            # print(output.shape, output_i.shape)
            # print('## debug msg: output is:', output.shape)
            # print(output)
            output[:,i,:] = output_i.view(output_i.shape[0], output_i.shape[2])   
        return output


class AttentionLayer(nn.Module):
    def __init__(self, d_word_embedding, 
        d_h, d_s, d_v):
        super(AttentionLayer, self).__init__()
        self.attention_s = nn.Linear(d_s, d_v)
        self.attention_h = nn.Linear(2*d_h, d_v)
        self.attention = nn.Linear(d_v, 1)
        self.d_s=d_s

    def forward(self, s_prev, h, src_max_sent_len):
        # Shape of s_prev = batch_size x d_s
        # Shape of s_prev after transformation = batch_size x src_max_sent_len x d_s
        s_prev = s_prev.reshape(h.shape[0], -1, self.d_s).expand(h.shape[0], src_max_sent_len, self.d_s)
        e_i = self.attention(nn.Tanh()(self.attention_s(s_prev)+
                        self.attention_h(h)))
        # print(e_i.shape)
        attention = nn.Softmax(dim=1)(e_i)
        return attention


class DeRNN(nn.Module):
    def __init__(self, d_h, d_s, d_word_embedding, tgt_vocab_size, d_l):
        super(DeRNN, self).__init__()

        self.t_s_ = nn.Linear(d_s, 2*d_l)
        self.t_y_ = nn.Linear(d_word_embedding, 2*d_l)
        self.t_c_ = nn.Linear(2*d_h, 2*d_l)
        self.maxpool = nn.MaxPool1d(2)
        self.output = nn.Linear(d_l, tgt_vocab_size)

        self.word_embed_layer = nn.Embedding(tgt_vocab_size, d_word_embedding)
        self.z_i_y = nn.Linear(d_word_embedding, d_s)
        self.z_i_s = nn.Linear(d_s, d_s)
        self.z_i_c = nn.Linear(2*d_h, d_s)
        self.r_i_y = nn.Linear(d_word_embedding, d_s)
        self.r_i_s = nn.Linear(d_s, d_s)
        self.r_i_c = nn.Linear(2*d_h, d_s)
        self.s_tilde_y = nn.Linear(d_word_embedding, d_s)
        self.s_tilde_s = nn.Linear(d_s, d_s)
        self.s_tilde_c = nn.Linear(2*d_h, d_s)



    def forward(self, s_prev, y_prev, c_i):
        # print(y_prev)
        word_embed = self.word_embed_layer(y_prev)
        # print(c_i)
        # y_prev = y_prev.float()
        # print(s_prev.shape, word_embed.shape, c_i.shape)
        t_tilde = self.t_s_(s_prev) + self.t_y_(word_embed) + self.t_c_(c_i)
        t = self.maxpool(t_tilde)
        # print(t.shape, t_tilde.shape)
        # output = nn.Softmax(dim=2)(self.output(t))
        output = self.output(t)

        z_i = nn.Sigmoid()(self.z_i_y(word_embed)+
                           self.z_i_s(s_prev)+
                           self.z_i_c(c_i))
        r_i = nn.Sigmoid()(self.r_i_y(word_embed)+
                           self.r_i_s(s_prev)+
                           self.z_i_c(c_i))
        s_tilde = nn.Tanh()(self.s_tilde_y(word_embed)+
                            self.s_tilde_s(r_i * s_prev)+
                            self.s_tilde_c(c_i))
        s_i = (1-z_i)*s_prev + z_i*s_tilde

        return output, s_i


class Transformer(nn.Module):
    def __init__(self, d_word_embedding, d_h, d_s,
                 src_vocab_size, tgt_vocab_size, max_sent_len
                 ):
        super().__init__()
        self.encoderLayer = Encoder(d_word_embedding=d_word_embedding, d_h=d_h, src_vocab_size=src_vocab_size)
        self.decoderLayer = Decoder(d_h=d_h, d_s=d_s, d_word_embedding=d_word_embedding, 
            tgt_vocab_size=tgt_vocab_size, d_l=d_s)

    def forward(self, x, src_max_sent_len, tgt_max_sent_len):
        # print(x)
        encoder_output = self.encoderLayer(x)
        output = self.decoderLayer(encoder_output, src_max_sent_len, tgt_max_sent_len-1)
        return output
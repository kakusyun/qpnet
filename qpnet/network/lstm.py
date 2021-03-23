import torch.nn as nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        assert len(x.size()) == 4, 'the input tensor should be 4-d.'
        x_reshape = x.view(-1, x.size(-2), x.size(-1))
        y, _ = self.module(x_reshape)
        if self.batch_first:
            y = y.view(x.size(0), -1, y.size(-2), y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-2), y.size(-1))
        return y


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, permute_way=2):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True, dropout=0)
        self.td_bilstm = TimeDistributed(self.rnn, batch_first=True)
        self.layer_norm = nn.LayerNorm(2 * hidden_size)
        self.permute = permute_way

    def forward(self, input):
        """
        input : batch_samples, timesteps, sequence_len, input_size
        visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        if self.permute == 0:
            x = input.permute(0, 3, 2, 1).contiguous()
        elif self.permute == 1:
            x = input.permute(0, 2, 1, 3).contiguous()
        else:
            x = input

        x = self.td_bilstm(x)
        x = self.layer_norm(x)
        return x

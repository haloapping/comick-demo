import torch
from torch import nn
from pathlib import Path

class POSTagger(nn.Module):
    def __init__(
        self,
        pretrained,
        input_size=64,
        hidden_size=128,
        dropout=0,
        bias=True,
        num_layers=1,
        output_size=24,
        batch_first=True,
        bidirectional=True,
        init_wb_with_kaiming_normal=False
    ):
        super(POSTagger, self).__init__()
        
        self.pretrained = pretrained
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bias = bias
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional
                
        self.feature = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            bias = self.bias,
            dropout = self.dropout,
            num_layers = self.num_layers,
            batch_first = self.batch_first,
            bidirectional = self.bidirectional
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features=2 * self.hidden_size if self.bidirectional else self.hidden_size, out_features=self.output_size, bias=self.bias),
            nn.Softmax(dim=-1)
        )

        self.load_state_dict(torch.load(self.pretrained, map_location=torch.device('cpu')))
        
        if init_wb_with_kaiming_normal:
            self.init_wb()

    def init_wb(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.LSTM)):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.kaiming_normal_(param)
                    else:
                        nn.init.kaiming_normal_(param.reshape(1, -1))
        
    def forward(self, feature):
        output, _ = self.feature(feature, None)
        
        return self.classifier(output)
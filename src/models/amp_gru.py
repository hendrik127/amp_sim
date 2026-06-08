import torch.nn as nn

class AmpGRU(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=3, 
                 dropout=0.0):
        super().__init__()
        
        # linear layer before GRU for more features
        self.input_proj = nn.Conv1d(1, input_size, 1)
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
            batch_first=True
        )
        
        self.output_proj = nn.Linear(hidden_size, 1)
    
    def forward(self, x):

        x = x.transpose(1, 2)
        gru_out, hidden = self.gru(x)  
        y = self.output_proj(gru_out)
        y = y.transpose(1, 2)
        
        return y
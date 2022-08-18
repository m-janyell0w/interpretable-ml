# Build SNAM class
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

# NN architecture as used by Xu et al.
class SampleNet(nn.Module):
    
    def __init__(self,seed=1, m=100, n_output=1):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.fc0 = nn.Linear(1, 1,bias=False)
        self.fc1 = nn.Linear(1, m)
        self.fc2 = nn.Linear(m, 50)
        self.fc3 = nn.Linear(50, n_output, bias=False)

    def forward(self, x):
        x = F.relu(self.fc2(F.relu(self.fc1(x)))).to(self.device)
        return self.fc3(x)

    def name(self):
        return "CaliforniaNAM_FeatureNN"

# NN architecture class as used by Freyberger et al.
class PyramidNet(nn.Module):
    
    def __init__(self,seed=1, m=64, n_output=1):
        super().__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.fc0 = nn.Linear(1, 1,bias=False)
        self.fc1 = nn.Linear(1, m)
        self.fc2 = nn.Linear(m, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, n_output, bias=False)

    def forward(self, x):
        x = F.relu(self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))))).to(self.device)
        return self.fc5(x)

    def name(self):
        return "PyramidNAM_FeatureNN"

# Neural additive model architecture as proposed by Agarwal et al.
class NAM(nn.Module):
    
    def __init__(self, feature_size, output_size, ModelA, seed=1, m=100):
        super(NAM, self).__init__()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.feature_size = feature_size
        self.output_size = output_size
        self.feature_nns = torch.nn.ModuleList([
            ModelA(seed=seed, m=m, n_output=self.output_size)
            for i in range(self.feature_size)
        ])
        self.bias = nn.Parameter(torch.zeros(output_size), requires_grad=True)

        
    def forward(self, input):
        """
        Implements forward pass through, returning the output f(x) which is a sum
        and f_out, i.e. the learned shape functions f_j(x).
        Inputs:   X -> torch.tensor
        Outputs:  f(X) -> torch.tensor
                  f_out -> 
        """
        output=self.feature_nns[0](input[:, [0]])+self.bias # input layer
        for i in range(1,self.feature_size):                # hidden layers
            output+=self.feature_nns[i](input[:,[i]])       # add layer output to total output
        f_out=torch.cat([self.feature_nns[i](input[:, [i]]).reshape(len(input), -1, self.output_size)
         for i in range(self.feature_size)], dim=1)         # output one value per feature
        return [output, f_out]                              # total output is sum
                                                            # of f-outputs
    def name(self):
        return "CaliforniaNAM"
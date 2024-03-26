#Graph Convolutional Layer

import torch
import torch.nn as nn


class GCNlayer(nn.Module):

    def __init__(self, d_in, d_out):
        super(GCNlayer, self).__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)     #W is a trainable weight matrix
        self.activation = nn.ReLU()
        return

    def forward(self, A, X):

        # A: [batch_size, N, N]
        # X: [batch_size, N, F]
        # W: [F', F]

        # step1: Compute WX (linear transformation of input features)
        X = self.W(X)       # [batch_size, N, F']

        #step 2: normalize the features by multiplying with D^-1/2 * A * D^-1/2
        deg_matrix = torch.sum(A, dim=2)    # [batch_size, N]
        deg_matrix = torch.diag_embed(deg_matrix**(-1/2))  # [batch_size, N, N] 
        deg_matrix_norm = torch.bmm(deg_matrix, torch.bmm(A, deg_matrix))

        #step 3: Compute the output
        X = torch.bmm(deg_matrix_norm, X)   # [batch_size, N, F']

        #step 4: Apply activation function
        X = self.activation(X)  # [batch_size, N, F']

        return X

#N: Number of nodes in the graph
#F: Number of input features per node
#F': Number of output features per node
#A: Adjacency matrix, shape [N, N]
#X: Feature matrix, shape [N, F]
#W: Trainable weight matrix, shape [F, F']
d_in = 2            #F
d_out = 6          #F'


#Adjacency Matrix
A = torch.tensor([[0, 1, 0, 0],
                    [1, 0, 1, 0],
                    [0, 1, 0, 1],
                    [0, 0, 1, 0]], dtype=torch.float)
# Reshape adjacency matrix to include batch size
A = A.view(1, A.shape[0], A.shape[1])  # [batch_size, N, N]: Here batch_size = 1

#Feature Matrix
X = torch.tensor([[i, -i] for i in range(A.shape[1])], dtype=torch.float)
# Reshape feature matrix to include batch size
X = X.view(1, X.shape[0], X.shape[1])  # [batch_size, N, F]: Here batch_size = 1

gcn_layer = GCNlayer(X.shape[2], d_out)

with torch.no_grad():
    output = gcn_layer.forward(A, X)
    print("Output: ", output)
    print("Output shape: ", output.shape)



import torch
import torch.nn as nn

class GAT_layer(nn.Module):
    
    def __init__(self, d_in, d_out, num_heads):
        super(GAT_layer, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads

        self.weight = nn.Linear(self.d_in, self.d_out*num_heads, bias=False)    #W is a trainable weight matrix
        self.multi_head_attention = []
        for i in range(num_heads):
            self.multi_head_attention.append(nn.Linear(2*self.d_out, 1, bias=False))
        self.activation = nn.ReLU()
        self.activation_after_attention = nn.ReLU()
        return
    
    def split_heads(self, X):
        # X: [batch_size, N, F'*num_heads]
        # return: [batch_size, num_heads, N, F']
        X = X.view(X.shape[0], X.shape[1], self.num_heads, X.shape[2]//self.num_heads).permute(0, 2, 1, 3)
        return X

    def attention_mechanism(self, X, A):
        edges = A.nonzero(as_tuple=False) # Returns indices where the adjacency matrix is not 0 => edges
        num_nodes = A.shape[0]
        self.edge_indices_row = edges[:,0] * num_nodes + edges[:,1]
        self.edge_indices_col = edges[:,0] * num_nodes + edges[:,2]

        a_input=[]
        for i in range(self.num_heads):
            a_input.append(torch.cat((torch.index_select(input=X[:,i,:,:], dim=1, index=self.edge_indices_row), torch.index_select(input=X[:,i,:,:], dim=1, index=self.edge_indices_col)), dim=-1))

        a_input = torch.cat(a_input, dim=2) 

        a_input = a_input.permute(1, 0, 2)
        a_input = self.split_heads(a_input)
        a_input = a_input.permute(1, 0, 2, 3)

        a_out = []
        for i in range(self.num_heads):
            attention = self.activation(self.multi_head_attention[i](a_input[i])).view(a_input[i].shape[0], -1)   
            attention = torch.softmax(attention, dim=0)
            a_out.append(attention)

        a_out = torch.stack(a_out, dim=0)
        a_out = a_out.permute(2, 0, 1)
        return a_out
    
    def forward(self, A, X, concat):
        out = self.weight(X)    # [batch_size, N, F'*num_heads]
        out = self.split_heads(out) # [batch_size, num_heads, N, F']
        multi_head_attention_vals = self.attention_mechanism(out, A)

        output = []

        for head in range(self.num_heads):
            attention_matrix = torch.zeros(X.shape[0], A.shape[1], A.shape[1])
            for id,(i,j) in enumerate(zip(self.edge_indices_row, self.edge_indices_col)):
                attention_matrix[:, i.item(), j.item()] = multi_head_attention_vals[:, head, id]
            #print("Attention Matrix: ", attention_matrix)

            node_feat = out[:, head, :, :].view(X.shape[0], out.shape[2], out.shape[3])
            #print("Node Features: ", node_feat)

            output.append(self.activation_after_attention(attention_matrix@node_feat))

        if concat:
            output = torch.cat(output, dim=-1)
            
        else:
            #convert list to tensor
            output = torch.stack(output, dim=0).permute(1, 0, 2,3)
            output = output.mean(dim=1)

        return output




#N: Number of nodes in the graph
#F: Number of input features per node
#F': Number of output features per node
#A: Adjacency matrix, shape [N, N]
#X: Feature matrix, shape [N, F]
#W: Trainable weight matrix, shape [F, F']
d_in = 2            #F
d_out = 5          #F'
num_heads = 3


#Adjacency Matrix
A = torch.tensor([[0, 1, 0, 0, 1],
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 0],
                    [1, 0 ,0, 0, 0]], dtype=torch.float)
# Reshape adjacency matrix to include batch size
A = A.view(1, A.shape[0], A.shape[1])  # [batch_size, N, N]: Here batch_size = 1

#Feature Matrix
X = torch.tensor([[i, -i] for i in range(A.shape[1])], dtype=torch.float)
# Reshape feature matrix to include batch size
X = X.view(1, X.shape[0], X.shape[1])  # [batch_size, N, F]: Here batch_size = 1

gcn_layer = GAT_layer(X.shape[2], d_out, num_heads)

with torch.no_grad():
    output = gcn_layer.forward(A, X, concat=True)
    print("Output: ", output)
    #print("Output shape: ", output.shape)
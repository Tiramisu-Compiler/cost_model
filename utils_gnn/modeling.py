from unicodedata import bidirectional
import torch
from torch import nn
from .data_utils import MAX_NUM_TRANSFORMATIONS, MAX_TAGS
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import DataLoader
from torch_geometric.data import Data

def initialization_function_sparse(x):
    return nn.init.sparse_(x, sparsity=0.1)

def initialization_function_xavier(x):
    return nn.init.xavier_uniform_(x)

# Define the architecture of the cost model
class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=1,
        lstm_embedding_size=100,
        expr_embed_size=100,
        loops_tensor_size=8,
        device="cpu",
        num_layers=1,
        bidirectional=True,
    ):
        super().__init__()
        self.device = device
        embedding_size = comp_embed_layer_sizes[-1]
        
        regression_layer_sizes = [embedding_size] + comp_embed_layer_sizes[-2:]
        concat_layer_sizes = [
            embedding_size * 2 + loops_tensor_size
        ] + comp_embed_layer_sizes[-2:]
        
        comp_embed_layer_sizes = [
            input_size + lstm_embedding_size * (2 if bidirectional else 1) * num_layers + expr_embed_size
        ] + comp_embed_layer_sizes
        
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        
        # Create the transformation encoding layers
        self.encode_vectors = nn.Linear(
            MAX_TAGS,
            MAX_TAGS,
            bias=True,
        )
        # Create the computation embedding layers
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(
                    comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True
                )
            )
            initialization_function_xavier(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        # Create the final regression layers
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(
                    regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True
                )
            )
            initialization_function_xavier(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
            
        # Create the feed forward netwrok responsible for embedding loop levels
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            initialization_function_xavier(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        # Output layer
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        
        
        initialization_function_xavier(self.predict.weight)
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)
        # Initialize a tensor to represent the absence of computations at a level in the program tree
        self.no_comps_tensor = nn.Parameter(
            initialization_function_xavier(torch.zeros(1, embedding_size))
        )
        # Initialize a tensor to represent the absence of child loops at a level in the program tree
        self.no_nodes_tensor = nn.Parameter(
            initialization_function_xavier(torch.zeros(1, embedding_size))
        )
        # LSTM to encode computations
        self.comps_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        # LSTM to encode child loop levels
        self.nodes_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        # LSTM to encode program roots
        self.roots_lstm = nn.LSTM(
            comp_embed_layer_sizes[-1], embedding_size, batch_first=True
        )
        # LSTM to encode computations
        self.transformation_vectors_embed = nn.LSTM(
            MAX_TAGS,
            lstm_embedding_size,
            batch_first=True,
            bidirectional=bidirectional,
            num_layers=num_layers,
        )
        # LSTM to encode computation expressions
        self.exprs_embed = nn.LSTM(
            11,
            expr_embed_size,
            batch_first=True,
        )
    # Recursive function to embed a root of the program
    def get_hidden_state(self, node, comps_embeddings, loops_tensor):
        nodes_list = []
        for n in node["child_list"]:
            # Recusrive call to embed all the children of the loop first if they exist
            nodes_list.append(self.get_hidden_state(
                n, comps_embeddings, loops_tensor))
        
        if nodes_list != []:
            # Pass the embedding of all the child loops through the nodes LSTM
            nodes_tensor = torch.cat(nodes_list, 1)
            lstm_out, (nodes_h_n, nodes_c_n) = self.nodes_lstm(nodes_tensor)
            nodes_h_n = nodes_h_n.permute(1, 0, 2)
        
        else: # If there are no child loops contained within this level
            # The nodes embedding is a random vector (no_nodes_tensor) that represents that there are no nodes underneath this level
            nodes_h_n = torch.unsqueeze(self.no_nodes_tensor, 0).expand(
                comps_embeddings.shape[0], -1, -1
            )
        if node["has_comps"]:
            # If there are computations contained in this loop, pass them through the computations LSTM
            selected_comps_tensor = torch.index_select(
                comps_embeddings, 
                1, 
                node["computations_indices"].to(self.device)
            )
            lstm_out, (comps_h_n, comps_c_n) = self.comps_lstm(
                selected_comps_tensor
            )
            comps_h_n = comps_h_n.permute(1, 0, 2)
        else: # If there are no child computations contained within this level
            # The computations embedding is a random vector (no_comps_tensor) that represents that there are no computations underneath this level
            comps_h_n = torch.unsqueeze(self.no_comps_tensor, 0).expand(
                comps_embeddings.shape[0], 
                -1, 
                -1
            )
        # Get the loop vector for this level
        selected_loop_tensor = torch.index_select(
            loops_tensor, 
            1, 
            node["loop_index"].to(self.device)
        )
        # Concatinate the loop vector, computations embedding and nodes (child loops) embedding
        x = torch.cat((nodes_h_n, comps_h_n, selected_loop_tensor), 2)
        # Pass the concatinated vector through a feed forward neural network
        for i in range(len(self.concat_layers)):
            x = self.concat_layers[i](x)
            x = self.concat_dropouts[i](self.ELU(x))
        return x

    def forward(self, tree_tensors):
        # Separate the input tensor
        tree, comps_tensor_first_part, comps_tensor_vectors, comps_tensor_third_part, loops_tensor, functions_comps_expr_tree = tree_tensors
        
        # Embed all the expressions in the tree 
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        
        # Expressions embedding layer
        x = functions_comps_expr_tree.view(batch_size* num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        
        # Embed all the computations in the tree
        batch_size, num_comps, __dict__ = comps_tensor_first_part.shape
        
        first_part = comps_tensor_first_part.to(self.device).view(batch_size * num_comps, -1)
        vectors = comps_tensor_vectors.to(self.device) # No need to reshape this tensor since we transformed it when loading the data
        third_part = comps_tensor_third_part.to(self.device).view(batch_size * num_comps, -1)
        
        # Pass the transformation vectors through the vector encoding LSTM
        vectors = self.encode_vectors(vectors)
        _, (prog_embedding, _) = self.transformation_vectors_embed(vectors)
        prog_embedding = prog_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        
        # Concatinate the leftover parts from the computatuion, the vectors embedding, and the expression embedding
        x = torch.cat(
            (
                first_part,
                prog_embedding,
                third_part,
                expr_embedding,
            ),
            dim=1,
        ).view(batch_size, num_comps, -1)
        
        # Pass the concatinated vector through a feed forward neural network to extract the final computation embedding vector
        for i in range(len(self.comp_embedding_layers)):
            x = self.comp_embedding_layers[i](x)
            x = self.comp_embedding_dropouts[i](self.ELU(x))
        comps_embeddings = x
        
        # For each root in the program tree structure 
        roots_list = []
        for root in tree["roots"]:
            # We call the recusrive embedding function to extract the root's embedding
            roots_list.append(
                self.get_hidden_state(
                    root, 
                    comps_embeddings, 
                    loops_tensor
                )
            )
        
        # We concatinate the roots and pass them through an LSTM
        roots_tensor = torch.cat(roots_list, 1)
        lstm_out, (roots_h_n, roots_c_n) = self.roots_lstm(roots_tensor)
        roots_h_n = roots_h_n.permute(1, 0, 2)
        
        # Finally, we pass the output of the roots LSTM to the regression layers
        x = roots_h_n
        for i in range(len(self.regression_layers)):
            x = self.regression_layers[i](x)
            x = self.regression_dropouts[i](self.ELU(x))
        out = self.predict(x)
        # We know the speedups to be predicted need to be greater or  equal to zero
        # We use the LeakyRelu to assure this while avoiding the dying ReLu problem
        return self.LeakyReLU(out[:, 0, 0])


class TransformerModel(nn.Module):
    def __init__(self, input_size, embedding_dim=256, nhead=8, num_layers=4, dropout=0.2, output_size=1):
        super().__init__()
        self.embedding = nn.Linear(input_size, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embedding_dim, output_size)
        self.activation = nn.LeakyReLU(0.01)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x[:, 0, :])  # Use the first token as the output
        return self.activation(x)
        

import torch.nn.functional as F

class SimpleGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, heads=4):
        """
        A simple GAT model that:
         - Uses two GATConv layers
         - Pools the node embeddings
         - Outputs a single scalar per graph (e.g. speedup or cost)
        """
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)

        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, concat=True)

        self.lin = nn.Linear(hidden_channels * heads, out_channels)

    def forward(self, data):
        """
        data is a PyG 'Data' object (or a mini-batch).
        Typically, it has:
          data.x          -> Node feature matrix [num_nodes, in_channels]
          data.edge_index -> Edge index [2, num_edges]
          data.batch      -> Graph index if multiple graphs in the batch
        """
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        x = self.gat1(x, edge_index)
        x = F.elu(x)   # or F.relu(x)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        # Final linear -> shape [batch_size, out_channels], then squeeze
        out = self.lin(x).squeeze(dim=-1)
        return out


class SimpleGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        """
        A simple GCN model that:
         - Uses two GCNConv layers
         - Pools the node embeddings
         - Outputs a single scalar per graph (e.g. speedup)
        """
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        """
        data is a PyG 'Data' object or a mini-batch from DataLoader.
        Typically, data will have:
          data.x          -> Node feature matrix [num_nodes, in_channels]
          data.edge_index -> Edge index [2, num_edges]
          data.batch      -> Optional batch assignment if multiple graphs are merged
          data.y          -> Speedup or cost label (not used directly in forward)
        """
        x, edge_index = data.x, data.edge_index

        if hasattr(data, 'batch'):
            batch = data.batch
        else:
            # single-graph scenario
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        out = self.lin(x).squeeze(dim=-1)
        # out has shape [batch_size], one scalar per graph
        return out
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GraphNorm, GlobalAttention

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=True),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.net(x)
        return self.drop(x)

class ResidualGIN(nn.Module):
    """
    Faster & stabler than your GAT baseline on big batches.
    - Residual GIN blocks with GraphNorm
    - JK (max) over layers
    - GlobalAttention pooling
    - Output head initialized to predict ~1.0 (good for MAPE on speedups)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        num_layers: int = 5,
        dropout: float = 0.2,
        out_channels: int = 1,
    ):
        super().__init__()
        assert num_layers >= 2

        self.input_lin = nn.Linear(in_channels, hidden_channels, bias=True)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            # Two-layer MLP inside GIN
            mlp = MLP(hidden_channels, hidden_channels, hidden_channels, dropout=dropout)
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(GraphNorm(hidden_channels))

        self.dropout = nn.Dropout(dropout)

        # Global attention pooling with a small gate MLP
        self.gate_nn = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
        )
        self.pool = GlobalAttention(self.gate_nn)

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        # Initialize final layer to predict ~1.0 at start (important for MAPE)
        nn.init.zeros_(self.head[-1].weight)
        with torch.no_grad():
            self.head[-1].bias.fill_(1.0)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.input_lin(x)

        # Residual GIN stack + GraphNorm
        xs = []
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, edge_index)
            h = norm(h, batch)
            h = F.relu(h, inplace=True)
            h = self.dropout(h)
            # Residual
            x = x + h
            xs.append(x)

        # JK (max) over layers
        h_all = torch.stack(xs, dim=0)           # [L, N, C]
        x = torch.amax(h_all, dim=0)             # [N, C]

        # Global pooling
        g = self.pool(x, batch)                   # [B, C]

        # Head (bias ~ 1.0 => sane start for MAPE)
        out = self.head(g).squeeze(-1)            # [B]
        return out

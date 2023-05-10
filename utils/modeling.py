from unicodedata import bidirectional
import torch
from torch import nn
from utils.data_utils import MAX_NUM_TRANSFORMATIONS, MAX_TAGS

# Define the architecture of the cost model
class Model_Recursive_LSTM_v2(nn.Module):
    def __init__(
        self,
        input_size,
        comp_embed_layer_sizes=[600, 350, 200, 180],
        drops=[0.225, 0.225, 0.225, 0.225],
        output_size=3,
        expr_embed_size=100,
        loops_tensor_size=2,
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
            input_size + expr_embed_size
        ] + comp_embed_layer_sizes
        
        self.comp_embedding_layers = nn.ModuleList()
        self.comp_embedding_dropouts = nn.ModuleList()
        self.regression_layers = nn.ModuleList()
        self.regression_dropouts = nn.ModuleList()
        self.concat_layers = nn.ModuleList()
        self.concat_dropouts = nn.ModuleList()
        
        # Create the computation embedding layers
        for i in range(len(comp_embed_layer_sizes) - 1):
            self.comp_embedding_layers.append(
                nn.Linear(
                    comp_embed_layer_sizes[i], comp_embed_layer_sizes[i + 1], bias=True
                )
            )
            nn.init.xavier_uniform_(self.comp_embedding_layers[i].weight)
            self.comp_embedding_dropouts.append(nn.Dropout(drops[i]))
        # Create the final regression layers
        for i in range(len(regression_layer_sizes) - 1):
            self.regression_layers.append(
                nn.Linear(
                    regression_layer_sizes[i], regression_layer_sizes[i + 1], bias=True
                )
            )
            nn.init.xavier_uniform_(self.regression_layers[i].weight)
            self.regression_dropouts.append(nn.Dropout(drops[i]))
            
        # Create the feed forward netwrok responsible for embedding loop levels
        for i in range(len(concat_layer_sizes) - 1):
            self.concat_layers.append(
                nn.Linear(concat_layer_sizes[i], concat_layer_sizes[i + 1], bias=True)
            )
            nn.init.xavier_uniform_(self.concat_layers[i].weight)
            nn.init.zeros_(self.concat_layers[i].weight)
            self.concat_dropouts.append(nn.Dropout(drops[i]))
        # Output layer
        self.predict = nn.Linear(regression_layer_sizes[-1], output_size, bias=True)
        
        
        nn.init.xavier_uniform_(self.predict.weight)
        self.ELU = nn.ELU()
        self.LeakyReLU = nn.LeakyReLU(0.01)
        self.Softmax = nn.Softmax(dim=0)
        # Initialize a tensor to represent the absence of computations at a level in the program tree
        self.no_comps_tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, embedding_size))
        )
        # Initialize a tensor to represent the absence of child loops at a level in the program tree
        self.no_nodes_tensor = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, embedding_size))
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
#     def approximate_output(continious_predictions):
        
        
    def forward(self, tree_tensors):
        # Separate the input tensor
        tree, comps_tensor, loops_tensor, functions_comps_expr_tree = tree_tensors
        
        # Embed all the expressions in the tree 
        batch_size, num_comps, len_sequence, len_vector = functions_comps_expr_tree.shape
        
        # Expressions embedding layer
        x = functions_comps_expr_tree.view(batch_size* num_comps, len_sequence, len_vector)
        _, (expr_embedding, _) = self.exprs_embed(x)
        
        expr_embedding = expr_embedding.permute(1, 0, 2).reshape(
            batch_size * num_comps, -1
        )
        
        # Embed all the computations in the tree
        batch_size, num_comps, __dict__ = comps_tensor.shape
        
        first_part = comps_tensor.to(self.device).view(batch_size * num_comps, -1)
        x = torch.cat(
            (
                first_part,
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
        
#         output = torch.cat(
#             (
#                 self.Softmax(out[:,0,:10]),
#                 torch.sigmoid(out[:,0,10:11]),
#                 torch.sigmoid(out[:,0,11:12]),
#                 torch.sigmoid(out[:,0,12:]),
#             ),
#             dim=-1
#         )
#         output = torch.cat(
#             (
#                 out[:,0,:10],
#                 torch.sigmoid(out[:,0,10:11]),
#                 torch.sigmoid(out[:,0,11:12]),
#                 torch.sigmoid(out[:,0,12:]),
#             ),
#             dim=-1
#         )
        output = torch.cat(
            (
                torch.sigmoid(out[:,0,0:1]),
                torch.sigmoid(out[:,0,1:2]),
                torch.sigmoid(out[:,0,2:]),
            ),
            dim=-1
        )
        return output
#         return self.LeakyReLU(out[:, 0, :])

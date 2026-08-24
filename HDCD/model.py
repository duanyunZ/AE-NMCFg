import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
from torch_scatter import scatter_add
from torch_geometric.utils import softmax, degree
from torch_geometric.nn.conv import MessagePassing, GCNConv
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.data import Data, Batch


class HypergraphConv(MessagePassing):
    def __init__(self, in_channels, out_channels, training=True, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.training = training
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j.view(-1, self.heads, self.out_channels)
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out
        return out

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        x = torch.matmul(x, self.weight)
        alpha = None
        if self.use_attention:
            x = x.view(-1, self.heads, self.out_channels)
            x_i, x_j = x[hyperedge_index[0]], x[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        if hyperedge_weight is None:
            D = degree(hyperedge_index[0], x.size(0), x.dtype)
        else:
            D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                            hyperedge_index[0], dim=0, dim_size=x.size(0))
        D = 1.0 / D
        D[D == float("inf")] = 0

        if hyperedge_index.numel() == 0:
            num_edges = 0
        else:
            num_edges = hyperedge_index[1].max().item() + 1
        B = 1.0 / degree(hyperedge_index[1], num_edges, x.dtype)
        B[B == float("inf")] = 0
        if hyperedge_weight is not None:
            B = B * hyperedge_weight

        num_nodes = x.size(0)
        dif = max([num_nodes, num_edges]) - num_nodes
        x_help = F.pad(x, (0, 0, 0, dif))
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x_help, norm=B, alpha=alpha)
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        out = out[:num_nodes]
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out = out + self.bias
        return out


class GraphRepresentation(nn.Module):
    def __init__(self, args):
        super(GraphRepresentation, self).__init__()
        self.num_node_features = args.node_features
        self.num_edge_features = args.edge_features
        self.nhid = args.num_hidden
        self.enhid = args.num_hidden

    def DHT(self, edge_index, batch, add_loops=True):
        num_edge = edge_index.size(1)
        device = edge_index.device
        edge_to_node_index = torch.arange(0, num_edge, 1, device=device).repeat_interleave(2).view(1, -1)
        hyperedge_index = edge_index.T.reshape(1, -1)
        hyperedge_index = torch.cat([edge_to_node_index, hyperedge_index], dim=0).long()
        if add_loops:
            max_edge = hyperedge_index[1].max()
            loops = torch.cat([
                torch.arange(0, num_edge, 1, device=device).view(1, -1),
                torch.arange(max_edge + 1, max_edge + num_edge + 1, 1, device=device).view(1, -1)
            ], dim=0)
            hyperedge_index = torch.cat([hyperedge_index, loops], dim=1)
        return hyperedge_index, None


class Conv(GraphRepresentation):
    def __init__(self, args):
        super(Conv, self).__init__(args)
        self.args = args
        self.convs = self.get_convs()
        self.hyperconvs = self.get_convs(conv_type='Hyper')

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for i in range(self.args.num_convs):
            hyperedge_index, _ = self.DHT(edge_index, batch)
            x = F.relu(self.convs[i](x, edge_index))
            edge_attr = F.relu(self.hyperconvs[i](edge_attr, hyperedge_index))
        return x, edge_attr

    def get_convs(self, conv_type='GCN'):
        convs = nn.ModuleList()
        for i in range(self.args.num_convs):
            if conv_type == 'GCN':
                conv = GCNConv(self.num_node_features if i == 0 else self.nhid,
                               1 if i == self.args.num_convs - 1 else self.nhid, normalize=False)
            else:
                conv = HypergraphConv(self.num_edge_features if i == 0 else self.enhid,
                                      1 if i == self.args.num_convs - 1 else self.enhid)
            convs.append(conv)
        return convs


# HDCD ablation version: DIRECTED GRAPH only
class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        # Keep only: knowledge nodes + directed edges
        self.h_mu = nn.Parameter(torch.randn(args.student_n, args.knowledge_n, args.node_features))
        self.h_logvar = nn.Parameter(torch.randn(args.student_n, args.knowledge_n, args.node_features))
        self.e_d_mu = nn.Parameter(torch.randn(args.student_n, args.d_edge_n, args.edge_features))
        self.e_d_logvar = nn.Parameter(torch.randn(args.student_n, args.d_edge_n, args.edge_features))

        # Exercise embedding: keep only directed edge difficulty
        self.k_difficulty = nn.Embedding(args.exer_n, args.knowledge_n)
        self.e_d_difficulty = nn.Embedding(args.exer_n, args.d_edge_n)
        self.e_discrimination = nn.Embedding(args.exer_n, 1)

        # Directed graph convolution only
        self.d_gcn = Conv(args)

        # Input dimension: knowledge + directed edge
        self.fc1 = nn.Linear(args.knowledge_n + args.d_edge_n, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_out = nn.Linear(32, 1)

    def _reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _kl_divergence(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

    def forward(self, stu_id, exer_id, q_node, q_dedges, edge_index_directed):
        # Reparameterization: keep only knowledge nodes + directed edges
        h_raw = self._reparam(self.h_mu[stu_id], self.h_logvar[stu_id])
        e_d_raw = self._reparam(self.e_d_mu[stu_id], self.e_d_logvar[stu_id])

        # Apply sigmoid to get mastery
        h = torch.sigmoid(h_raw)
        e_d = torch.sigmoid(e_d_raw)

        # Exercise attributes
        k_diff = torch.sigmoid(self.k_difficulty(exer_id))
        e_d_diff = torch.sigmoid(self.e_d_difficulty(exer_id))
        e_disc = torch.sigmoid(self.e_discrimination(exer_id)) * 3.0

        # Directed graph representation
        data_list_d = []
        for i in range(stu_id.shape[0]):
            data_d = Data(x=h[i], edge_index=edge_index_directed, edge_attr=e_d[i])
            data_list_d.append(data_d)
        d_graph = Batch.from_data_list(data_list_d)
        h_d, e_d_upd = self.d_gcn(d_graph)

        h_d = h_d.reshape(stu_id.size(0), -1)
        e_d_upd = e_d_upd.reshape(stu_id.size(0), -1)

        # Fusion: knowledge + directed edges only
        input_node = (h_d - k_diff) * q_node * e_disc
        input_dedge = (e_d_upd - e_d_diff) * q_dedges * e_disc
        ecs = torch.cat([input_node, input_dedge], dim=-1)

        # MLP prediction
        x = F.relu(self.fc1(ecs))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)

        # KL regularization
        KL = (self._kl_divergence(self.h_mu[stu_id], self.h_logvar[stu_id]) +
              self._kl_divergence(self.e_d_mu[stu_id], self.e_d_logvar[stu_id]))

        # Return student knowledge proficiency for PR metric calculation
        # h shape: [batch_size, knowledge_n, node_features]
        # Average over node_features dimension to get comprehensive mastery for each knowledge
        stu_knowledge_pro = h.mean(dim=-1)  # [batch_size, knowledge_n]

        return logits, KL, stu_knowledge_pro
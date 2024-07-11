import torch.nn as nn
import hypll.nn as hnn
from hypll.tensors import TangentTensor

from networks.nets import DeepSet


def manifold_map(x, manifold):
    """
    Maps a tensor in Euclidean space onto a Riemannian Manifold
    """
    tangents = TangentTensor(x, man_dim=-1, manifold=manifold)
    return manifold.expmap(tangents)

class HyperbolicMLP(nn.Module):
    """
    MLP with 2 Euclidean layers and 2 hyperbolic layers
    in_features: input dimension
    manifold: Desired manifold (e.g. Poincare Disc)
    euc_width: Width of the Euclidean hidden layer
    hyp_width: Width of the hyperbolic hidden layer
    out_features: Output dimension
    """
    def __init__(self, in_features, out_features, manifold, euc_width=64, hyp_width=64):
        super(HyperbolicMLP, self).__init__()
        self.manifold = manifold

        # Euclidean layers
        self.euc_layer1 = nn.Linear(in_features, euc_width)
        self.euc_layer2 = nn.Linear(euc_width, hyp_width)
        self.euc_relu = nn.ReLU()

        # Hyperbolic layers
        self.hyp_layer1 = hnn.HLinear(in_features=hyp_width, out_features=hyp_width, manifold=manifold)
        self.hyp_layer2 = hnn.HLinear(in_features=hyp_width, out_features=out_features, manifold=manifold)
        self.hyp_relu = hnn.HReLU(manifold=manifold)

    def forward(self, x):
        """
        Forward pass.
        """
        # Pass through Euclidean layers
        x = self.euc_relu(self.euc_layer1(x))
        x = self.euc_layer2(x)

        # Map to hyperbolic space
        x = manifold_map(x, self.manifold)

        # Pass through Hyperbolic layers
        x = self.hyp_relu(self.hyp_layer1(x))
        x = self.hyp_layer2(x)
        return x
    

class HyperbolicDeepSet(nn.Module):
    """
    DeepSet architecture for set-valued inputs. Final layers transform input into hyperbolic space
    See: https://arxiv.org/abs/1703.06114
    """
    def __init__(self, input_dim, hidden_dim, output_dim, manifold):
        super(HyperbolicDeepSet, self).__init__()

        self.deepset = DeepSet(input_dim, hidden_dim, hidden_dim)
        self.manifold = manifold

        # Hyperbolic layers
        self.hyp_layer1 = hnn.HLinear(
            in_features=hidden_dim, out_features=hidden_dim, manifold=manifold
        )
        self.hyp_relu = hnn.HReLU(manifold=manifold)
        self.hyp_layer2 = hnn.HLinear(
            in_features=hidden_dim, out_features=output_dim, manifold=manifold
        )

    def forward(self, x, mask):
        # Process input through DeepSet
        x = self.deepset(x, mask)

        # Map directly to hyperbolic space
        x = manifold_map(x, self.manifold)

        # Pass through Hyperbolic layers
        x = self.hyp_layer1(x)
        x = self.hyp_relu(x)
        x = self.hyp_layer2(x)

        return x

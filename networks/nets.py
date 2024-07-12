import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the StateActionEncoder
class SmallEncoder(nn.Module):
    """
    Encodes state into a vector of size embedding_dim
    """
    def __init__(self, input_dim, embedding_dim):
        super(SmallEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)  # New layer
        self.fc4 = nn.Linear(64, embedding_dim)

    def forward(self, x):
        """
        Forward pass.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))  # New layer
        x = self.fc4(x)
        return x

class CategoricalMLP(nn.Module):
    def __init__(self, cat_features, embedding_dims, hidden_dims, output_dim):
        super(CategoricalMLP, self).__init__()
        
        # Embedding layers
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, emb_dim) 
            for num_categories, emb_dim in zip(cat_features, embedding_dims)
        ])
        
        # Calculate total embedding dimension
        total_emb_dim = sum(embedding_dims)
        
        # Fully connected layers
        layers = []
        input_dim = total_emb_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        # x shape: [..., num_features]
        original_shape = x.shape
        num_features = original_shape[-1]
        
        # Reshape to [-1, num_features]
        x_flat = x.view(-1, num_features)
        
        # Apply embeddings
        embeddings = [emb_layer(x_flat[:, i]) for i, emb_layer in enumerate(self.embeddings)]
        concat_embeddings = torch.cat(embeddings, dim=1)
        
        # Pass through MLP
        output = self.mlp(concat_embeddings)
        
        # Reshape output back to original batch shape + [output_dim]
        return output.view(*original_shape[:-1], -1)

class LabelEncoder(nn.Module):
    """
    Encodes a categorical input into a vector of size embedding_dim
    """
    def __init__(self, num_categories, embedding_dim):
        super(LabelEncoder, self).__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, embedding_dim)

    def forward(self, categorical_data):
        """
        Forward pass.
        """
        embedded = self.embedding(categorical_data)
        # print(embedded)
        x = F.relu(self.fc1(embedded))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DeepSet(nn.Module):
    """
    DeepSet architecture for set-valued inputs.
    See: https://arxiv.org/abs/1703.06114
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DeepSet, self).__init__()

        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Correct calculation for upper triangle size
        cov_dim = (hidden_dim * (hidden_dim + 1)) // 2
        
        # Separate network for processing covariance
        self.cov_network = nn.Sequential(
            nn.Linear(cov_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.rho = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Now 2 * hidden_dim due to mean and processed covariance
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, mask):
        # x shape: (batch_size, max_set_size, input_dim)
        # mask shape: (batch_size, max_set_size)
        # print(f'x shape: {x.shape}, mask shape: {mask.shape}')
        # Apply phi to each element
        x = self.phi(x)  # Shape: (batch_size, max_set_size, hidden_dim)

        # Apply mask to zero out padding
        x = x * mask.unsqueeze(-1)

        # Compute mean
        x_sum = torch.sum(x, dim=1)
        set_sizes = torch.sum(mask, dim=1, keepdim=True)
        x_mean = x_sum / set_sizes  # Shape: (batch_size, hidden_dim)

        # Compute covariance
        x_centered = x - x_mean.unsqueeze(1)
        x_centered = x_centered * mask.unsqueeze(-1)  # Apply mask again
        cov = torch.bmm(x_centered.transpose(1, 2), x_centered) / (
            set_sizes - 1
        ).unsqueeze(-1).clamp(min=1)

        # Extract upper triangular part of covariance (including diagonal)
        batch_size, hidden_dim, _ = cov.shape
        triu_indices = torch.triu_indices(hidden_dim, hidden_dim)
        cov_vector = cov[:, triu_indices[0], triu_indices[1]]

        # Process covariance vector through its own network
        processed_cov = self.cov_network(cov_vector)  # Shape: (batch_size, hidden_dim)

        # Concatenate mean and processed covariance information
        x_combined = torch.cat([x_mean, processed_cov], dim=1)

        # Apply rho to the combined representation
        output = self.rho(x_combined)

        return output

class PointCloudEncoder(nn.Module):
    """
    Encoder for a small PointCloud, can also be used for set-valued inputs
    """
    def __init__(self, input_dim=8, output_dim=256):
        super(PointCloudEncoder, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        self.linear1 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, output_dim)

    def forward(self, x, mask):
        # x shape: (batch_size, n, 8)
        x = x * mask.unsqueeze(-1)
        x = x.transpose(2, 1)  # (batch_size, 8, n)

        # Point-wise MLPs
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        # Max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256)

        # Fully connected layers
        x = F.relu(self.bn4(self.linear1(x)))
        x = self.linear2(x)

        return x

import torch
import torch.nn.functional as F

    
class SurfaceEdgeModelBaseline(torch.nn.Module):
    def __init__(self, 
                 image_channels=3, image_size=224, img_dim=238,
                 num_tickers = 4, ticker_dim=128, dropout=0.2,
                 hidden_dim=64, out_dim=1):
        super().__init__()

        ## IMAGE PROCESSING ## 
        self.conv1 = torch.nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc_image = torch.nn.Linear(128, img_dim)

        ## TICKER PROCESSING ##
        self.ticker_embedding = torch.nn.Embedding(num_tickers, ticker_dim) 

        ## COMBINED PROCESSING ## 
        # Added +10 to account for the new 'stats' vector
        count_in_features = img_dim + ticker_dim + 1 + 1 + 1 + 1 + 10
        # count_in_features = ticker_dim + 1 + 1 + 1 + 1 + 10
        self.layer1 = torch.nn.Linear(count_in_features, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, 
        image, 
        tau,
        log_moneyness,
        is_call,
        mark,
        stats, # <-- NEW: The 10-dim vector
        ticker 
    ): 
        ## IMAGE PROCESSING ## 
        x_image = F.relu(self.conv1(image))
        x_image = self.pool1(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = F.relu(self.conv3(x_image))
        x_image = self.pool3(x_image)
        
        x_image = self.global_pool(x_image) 
        x_image = torch.flatten(x_image, 1) 
        x_image = F.relu(self.fc_image(x_image))

        ## TICKER PROCESSING ## 
        x_ticker = self.ticker_embedding(ticker) 

        ## SCALAR PROCESSING ##
        tau = tau.view(-1, 1)
        log_moneyness = log_moneyness.view(-1, 1)
        is_call = is_call.view(-1, 1).float() 
        mark = mark.view(-1, 1)
        # 'stats' is already [Batch, 10], so we don't need to view(-1, 1) it.

        ## CONCATENATION ##
        features = torch.cat((x_image, x_ticker, tau, log_moneyness, is_call, mark, stats), dim=1)
        # features = torch.cat((x_ticker, tau, log_moneyness, is_call, mark, stats), dim=1)

        ## FINAL MLP ##
        out_1 = self.layer1(features)
        out_1 = self.norm(out_1)
        out_1 = F.relu(out_1)
        out_1 = self.dropout(out_1)
        
        out_2 = self.out(out_1)

        return out_2



class ResidualBlock(torch.nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.LayerNorm(dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
            torch.nn.LayerNorm(dim)
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # residuals: x + block(x)
        return self.relu(x + self.block(x))


class SurfaceEdgeModelDeepHead(torch.nn.Module):
    def __init__(self, 
                 image_channels=3, image_size=224, img_dim=238,
                 num_tickers = 4, ticker_dim=128, dropout=0.2,
                 hidden_dim=64, out_dim=1):
        super().__init__()

        ## IMAGE PROCESSING ## 
        self.conv1 = torch.nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc_image = torch.nn.Linear(128, img_dim)

        ## TICKER PROCESSING ##
        self.ticker_embedding = torch.nn.Embedding(num_tickers, ticker_dim) 

        ## COMBINED PROCESSING ## 
        # Added +10 to account for the new 'stats' vector
        count_in_features = img_dim + ticker_dim + 1 + 1 + 1 + 1 + 10
        # count_in_features = ticker_dim + 1 + 1 + 1 + 1 + 10
        
        self.layer1 = torch.nn.Linear(count_in_features, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(hidden_dim)

        self.proj_to_hidden = torch.nn.Sequential(
            torch.nn.Linear(count_in_features, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.ReLU()
        )

        
        # Each residual block contains 2 internal layers
        self.res_block1 = ResidualBlock(hidden_dim, dropout)
        self.res_block2 = ResidualBlock(hidden_dim, dropout)

        self.out = torch.nn.Linear(hidden_dim, out_dim)


    def forward(self, 
        image, 
        tau,
        log_moneyness,
        is_call,
        mark,
        stats,
        ticker 
    ): 
        ## IMAGE PROCESSING ## 
        x_image = F.relu(self.conv1(image))
        x_image = self.pool1(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = F.relu(self.conv3(x_image))
        x_image = self.pool3(x_image)
        
        x_image = self.global_pool(x_image) 
        x_image = torch.flatten(x_image, 1) 
        x_image = F.relu(self.fc_image(x_image))

        ## TICKER PROCESSING ## 
        x_ticker = self.ticker_embedding(ticker) 

        ## SCALAR PROCESSING ##
        tau = tau.view(-1, 1)
        log_moneyness = log_moneyness.view(-1, 1)
        is_call = is_call.view(-1, 1).float() 
        mark = mark.view(-1, 1)
        # 'stats' is already [Batch, 10], so we don't need to view(-1, 1) it.

        ## CONCATENATION ##
        features = torch.cat((x_image, x_ticker, tau, log_moneyness, is_call, mark, stats), dim=1)
        # features = torch.cat((x_ticker, tau, log_moneyness, is_call, mark, stats), dim=1)

        ## FINAL MLP ##
        # layer 1: no dropout, projects to the hidden dimension 
        out_1 = self.proj_to_hidden(features)

        # layer 2-3
        out_3 = self.res_block1(out_1)
        # layer 4-5
        out_5 = self.res_block2(out_3)
        
        out_6 = self.out(out_5)

        return out_6


class SurfaceSequenceModel(torch.nn.Module):
    def __init__(self,
                 seq_len=25,                 # <-- NEW: Sequence length
                 image_channels=3, image_size=224, img_dim=238,
                 num_tickers=104, ticker_dim=128, dropout=0.2,
                 embed_dim=256, num_heads=4, num_layers=2, # <-- NEW: Transformer hyperparams
                 hidden_dim=64, out_dim=1):
        super().__init__()
        self.seq_len = seq_len

        ## IMAGE PROCESSING (Unchanged) ##
        self.conv1 = torch.nn.Conv2d(in_channels=image_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc_image = torch.nn.Linear(128, img_dim)

        ## TICKER PROCESSING (Unchanged) ##
        self.ticker_embedding = torch.nn.Embedding(num_tickers, ticker_dim)

        ## COMBINED FEATURE DIMENSION ##
        # count_in_features = img_dim + ticker_dim + 1 + 1 + 1 + 1 + 10 Now includes the label ONLY FOR HISTORICAL 
        count_in_features = img_dim + ticker_dim + 1 + 1 + 1 + 1 + 11

        ## SEQUENCE PROCESSING (NEW) ##
        # 1. Project the concatenated features to the Transformer's hidden dimension
        self.input_proj = torch.nn.Linear(count_in_features, embed_dim)
        
        # 2. Positional Encoding (Crucial for the model to understand time flow)
        self.pos_encoder = torch.nn.Parameter(torch.randn(1, seq_len, embed_dim))
        
        # 3. Self-Attention Transformer Layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True, 
            dropout=dropout
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        ## FINAL MLP ##
        # Note: Now takes embed_dim instead of count_in_features
        self.layer1 = torch.nn.Linear(embed_dim, hidden_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.out = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self,
        image,          # Shape: [Batch, SeqLen, Channels, H, W]
        tau,            # Shape: [Batch, SeqLen]
        log_moneyness,  # Shape: [Batch, SeqLen]
        is_call,        # Shape: [Batch, SeqLen]
        mark,           # Shape: [Batch, SeqLen]
        stats,          # Shape: [Batch, SeqLen, 10]
        ticker,         # Shape: [Batch]  (Constant per sequence)
        padding_mask=None # <-- NEW: Shape [Batch, SeqLen]. True for padding days, False for real days
    ):
        B, S = image.shape[0], image.shape[1]

        ## IMAGE PROCESSING ##
        # Flatten Batch and Sequence dimensions to push through CNN at once
        C, H, W = image.shape[2], image.shape[3], image.shape[4]
        flat_image = image.view(B * S, C, H, W)

        x_image = F.relu(self.conv1(flat_image))
        x_image = self.pool1(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = F.relu(self.conv3(x_image))
        x_image = self.pool3(x_image)

        x_image = self.global_pool(x_image)
        x_image = torch.flatten(x_image, 1)
        x_image = F.relu(self.fc_image(x_image))

        # Unflatten back to sequence format
        x_image = x_image.view(B, S, -1)  # Shape: [Batch, SeqLen, img_dim]

        ## TICKER PROCESSING ##
        x_ticker = self.ticker_embedding(ticker) # Shape: [Batch, ticker_dim]
        # Expand ticker across the sequence dimension
        x_ticker = x_ticker.unsqueeze(1).expand(-1, S, -1) # Shape: [Batch, SeqLen, ticker_dim]

        ## SCALAR PROCESSING ##
        # Unsqueeze the last dimension to allow concatenation
        tau = tau.unsqueeze(-1)                     # Shape: [Batch, SeqLen, 1]
        log_moneyness = log_moneyness.unsqueeze(-1) # Shape: [Batch, SeqLen, 1]
        is_call = is_call.unsqueeze(-1).float()     # Shape: [Batch, SeqLen, 1]
        mark = mark.unsqueeze(-1)                   # Shape: [Batch, SeqLen, 1]
        # 'stats' is already [Batch, SeqLen, 10]

        ## CONCATENATION ##
        # Cat all features along the last dimension (dim=2 or -1)
        features = torch.cat((x_image, x_ticker, tau, log_moneyness, is_call, mark, stats), dim=-1)

        ## SEQUENCE PROCESSING (NEW) ##
        # 1. Project to Transformer dimension
        x_seq = self.input_proj(features)
        
        # 2. Add Positional Encoding
        x_seq = x_seq + self.pos_encoder
        
        # 3. Apply Self-Attention
        # The padding mask ensures attention is not paid to fake/padded days for young contracts
        x_seq = self.transformer(x_seq, src_key_padding_mask=padding_mask)

        ## FINAL HEAD ##
        # Extract the representation of the *last* day in the sequence (Day t) to predict Day t+1
        x_last = x_seq[:, -1, :] # Shape: [Batch, embed_dim]

        out_1 = self.layer1(x_last)
        out_1 = self.norm(out_1)
        out_1 = F.relu(out_1)
        out_1 = self.dropout(out_1)

        out_2 = self.out(out_1)

        return out_2


import torch
import torch.nn as nn

class SurfaceSequenceTextModel(nn.Module):
    def __init__(self,
                 seq_len=25,
                 image_channels=3, img_dim=238,
                 num_tickers=104, ticker_dim=128, 
                 text_out_dim=64, # NEW: Text projection size
                 dropout=0.2,
                 embed_dim=256, num_heads=4, num_layers=2,
                 hidden_dim=64, out_dim=1):
        super().__init__()
        self.seq_len = seq_len

        ## 1. IMAGE PROCESSING (Strict Copy) ##
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_image = nn.Linear(128, img_dim)

        ## 2. TICKER PROCESSING (Strict Copy) ##
        self.ticker_embedding = nn.Embedding(num_tickers, ticker_dim)

        ## 3. TEXT PROCESSING (NEW) ##
        self.text_proj = nn.Linear(768, text_out_dim)

        ## 4. COMBINED FEATURE DIMENSION ##
        # img(238) + ticker(128) + scalars(4) + stats(11) + text(64) = 445
        count_in_features = img_dim + ticker_dim + 1 + 1 + 1 + 1 + 11 + text_out_dim

        ## 5. SEQUENCE PROCESSING (Strict Copy) ##
        self.input_proj = nn.Linear(count_in_features, embed_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        ## 6. FINAL MLP (Strict Copy) ##
        self.layer1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, image, tau, log_moneyness, is_call, mark, stats, ticker, text_emb, padding_mask=None):
        B, S = image.shape[0], image.shape[1]

        ## IMAGE PROCESSING ##
        C, H, W = image.shape[2], image.shape[3], image.shape[4]
        flat_image = image.view(B * S, C, H, W)

        x_image = F.relu(self.conv1(flat_image))
        x_image = self.pool1(x_image)
        x_image = F.relu(self.conv2(x_image))
        x_image = self.pool2(x_image)
        x_image = F.relu(self.conv3(x_image))
        x_image = self.pool3(x_image)

        x_image = self.global_pool(x_image)
        x_image = torch.flatten(x_image, 1)
        x_image = F.relu(self.fc_image(x_image))
        x_image = x_image.view(B, S, -1) 

        ## TICKER PROCESSING ##
        x_ticker = self.ticker_embedding(ticker)
        x_ticker = x_ticker.unsqueeze(1).expand(-1, S, -1)

        ## TEXT PROCESSING (NEW) ##
        # text_emb is [B, S, 768] -> [B, S, 64]
        x_text = F.relu(self.text_proj(text_emb))

        ## SCALAR PROCESSING ##
        tau = tau.unsqueeze(-1)
        log_moneyness = log_moneyness.unsqueeze(-1)
        is_call = is_call.unsqueeze(-1).float()
        mark = mark.unsqueeze(-1)

        ## CONCATENATION ##
        # Added x_text to the end of the feature vector
        features = torch.cat((x_image, x_ticker, tau, log_moneyness, is_call, mark, stats, x_text), dim=-1)

        ## SEQUENCE PROCESSING ##
        x_seq = self.input_proj(features)
        x_seq = x_seq + self.pos_encoder
        x_seq = self.transformer(x_seq, src_key_padding_mask=padding_mask)

        ## FINAL HEAD ##
        x_last = x_seq[:, -1, :] 

        out_1 = self.layer1(x_last)
        out_1 = self.norm(out_1)
        out_1 = F.relu(out_1)
        out_1 = self.dropout(out_1)

        return self.out(out_1)

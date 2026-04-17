import torch
import torch.nn.functional as F

    
class SurfaceEdgeModel(torch.nn.Module):
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

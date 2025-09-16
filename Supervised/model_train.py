import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ----- Dataset -----
class KnapsackDataset(Dataset):
    def __init__(self, path, max_items=20):
        with open(path, "r") as f:
            raw = json.load(f)
        self.data = raw
        self.max_items = max_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inst = self.data[idx]
        items = inst["items"]
        cap = inst["capacity"]
        sol = inst["solution"]

        # feature cho mỗi item: [weight, value, value/weight, capacity_norm]
        features = []
        for (w, v) in items:
            features.append([w, v, v/max(w,1), cap])
        
        # padding cho đủ max_items
        while len(features) < self.max_items:
            features.append([0,0,0,cap])
            sol.append(0)

        X = torch.tensor(features[:self.max_items], dtype=torch.float32)
        y = torch.tensor(sol[:self.max_items], dtype=torch.float32)
        return X, y

# ----- Model -----
class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden=64, max_items=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 1)
        self.max_items = max_items

    def forward(self, x):
        # x: (batch, n, d)
        b, n, d = x.shape
        x = x.view(b*n, d)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        out = torch.sigmoid(self.fc_out(h))
        out = out.view(b, n)
        return out


# ----- Training -----
def train():
    dataset = KnapsackDataset("data.json", max_items=20)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = MLP()
    opt = optim.Adam(model.parameters(), lr=5e-5)
    loss_fn = nn.BCELoss()

    for epoch in range(50):
        total_loss = 0
        for X, y in loader:
            opt.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "mlp_knapsack.pt")
    print("✅ Model saved: mlp_knapsack.pt")


if __name__ == "__main__":
    train()

import json
import torch
import torch.nn as nn
from dataset_gen import knapsack_dp

# ----- Model -----
class MLP(nn.Module):
    def __init__(self, input_dim=4, hidden=64, max_items=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_out = nn.Linear(hidden, 1)
        self.max_items = max_items

    def forward(self, x):
        b, n, d = x.shape
        x = x.view(b*n, d)
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        out = torch.sigmoid(self.fc_out(h))
        out = out.view(b, n)
        return out


def evaluate(model_path="mlp_knapsack.pt", data_path="data.json", max_items=20, n_samples=50):
    # Load dataset
    with open(data_path, "r") as f:
        dataset = json.load(f)

    # Load model
    model = MLP(max_items=max_items)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    correct_items = 0
    total_items = 0
    ratio_sum = 0
    n_eval = min(len(dataset), n_samples)

    for i in range(n_eval):
        inst = dataset[i]
        items = inst["items"]
        cap = inst["capacity"]
        sol = inst["solution"]

        # features
        features = []
        for (w, v) in items:
            features.append([w, v, v/max(w,1), cap])
        while len(features) < max_items:
            features.append([0,0,0,cap])
            sol.append(0)

        X = torch.tensor([features[:max_items]], dtype=torch.float32)

        with torch.no_grad():
            pred = model(X).squeeze(0).numpy()

        pred_bin = (pred > 0.5).astype(int).tolist()

        # item-level accuracy
        correct_items += sum(int(p == s) for p, s in zip(pred_bin, sol[:max_items]))
        total_items += max_items

        # value ratio
        weights = [w for (w, v) in items]
        values = [v for (w, v) in items]

        # DP value (optimal)
        opt_sol = knapsack_dp(weights, values, cap)
        opt_value = sum(v for (chosen, (w,v)) in zip(opt_sol, items) if chosen)

        # model value
        model_value = sum(v for (chosen, (w,v)) in zip(pred_bin, items) if chosen and w <= cap)
        ratio = model_value / opt_value if opt_value > 0 else 0
        ratio_sum += ratio

    print(f"✅ Item Accuracy: {correct_items/total_items:.2f}")
    print(f"✅ Avg Value Ratio: {ratio_sum/n_eval:.2f}")


if __name__ == "__main__":
    evaluate()

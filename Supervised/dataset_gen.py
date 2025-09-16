import random
import json

def knapsack_dp(weights, values, capacity):
    """Trả về vector chọn tối ưu bằng DP"""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weights[i-1]] + values[i-1])
            else:
                dp[i][w] = dp[i-1][w]

    # backtrack để tìm nghiệm
    res = [0] * n
    w = capacity
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i-1][w]:
            res[i-1] = 1
            w -= weights[i-1]
    return res


def gen_instance(n_items=15):
    weights = [random.randint(1, 100) for _ in range(n_items)]
    values = [random.randint(1, 100) for _ in range(n_items)]
    capacity = random.randint(int(0.4 * sum(weights)), int(0.6 * sum(weights)))

    solution = knapsack_dp(weights, values, capacity)
    return {
        "items": [[weights[i], values[i]] for i in range(n_items)],
        "capacity": capacity,
        "solution": solution
    }


if __name__ == "__main__":
    dataset = [gen_instance(n_items=random.randint(20, 50)) for _ in range(2000)]

    with open("data.json", "w") as f:
        json.dump(dataset, f)

    print("✅ Dataset generated: data.json")

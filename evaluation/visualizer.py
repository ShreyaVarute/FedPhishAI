import matplotlib.pyplot as plt
import json

def plot_privacy_tradeoff(results_path='experiments/results/privacy_tradeoff.json'):
    with open(results_path) as f:
        data = json.load(f)
    noise_levels = [d['noise'] for d in data]
    accuracies   = [d['accuracy'] for d in data]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, accuracies, 'b-o', linewidth=2, markersize=8)
    ax.fill_between(noise_levels, accuracies, alpha=0.1)
    ax.set_xlabel('DP Noise Scale (σ)', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Privacy–Utility Tradeoff Curve', fontsize=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments/results/privacy_tradeoff.png', dpi=150)
    plt.show()

def plot_fl_accuracy(history):
    """Plot accuracy per FL round from Flower history object."""
    rounds     = list(history.metrics_distributed.get('accuracy', {}).keys())
    accuracies = list(history.metrics_distributed.get('accuracy', {}).values())

    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracies, 'g-o', linewidth=2)
    plt.xlabel('FL Round')
    plt.ylabel('Accuracy')
    plt.title('Federated Learning Accuracy per Round')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('experiments/results/fl_accuracy.png', dpi=150)
    plt.show()

import os
import sys
import pickle
import matplotlib.pyplot as plt
import collections

# === 1. Load pickle file ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
pkl_file_path = os.path.join(os.path.dirname(__file__), '..', 'experiments', 'generations_20250508154230.pkl')

with open(pkl_file_path, 'rb') as f:
    data = pickle.load(f)

if not isinstance(data, list) or len(data) == 0:
    print("❌ Pickle data is not a non-empty list.")
    sys.exit(1)

record = data[0]
results = getattr(record, 'results', None)
if not results or not isinstance(results, list):
    print("❌ No valid 'results' list found.")
    sys.exit(1)

# === 2. Collect all numeric fields across results ===
field_values = collections.defaultdict(list)
for i, result in enumerate(results):
    if hasattr(result, '__dict__'):
        for key, value in result.__dict__.items():
            if isinstance(value, (int, float)):
                field_values[key].append(value)

# === 3. Plot all numeric fields ===
if not field_values:
    print("❌ No numeric fields found in results.")
else:
    generations = list(range(len(results)))
    plt.figure(figsize=(10, 6))

    for field, values in field_values.items():
        plt.plot(generations, values, marker='o', label=field)

    plt.xlabel("Generations")
    plt.ylabel("Value")
    plt.title("Metrics over Generations")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = os.path.join(os.path.dirname(__file__), 'all_metrics_plot.png')
    plt.savefig(output_path)
    print(f"✅ Plot saved to: {output_path}")

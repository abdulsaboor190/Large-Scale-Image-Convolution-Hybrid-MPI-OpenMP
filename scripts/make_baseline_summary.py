import csv
from collections import defaultdict

rows = []
with open('results/baseline_seq.csv', 'r') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

groups = defaultdict(list)
for r in rows:
    key = (r['filter'], int(r['k']))
    groups[key].append(float(r['compute_s']))

summary = []
for (filt, k), vals in sorted(groups.items()):
    vals.sort()
    med = vals[len(vals) // 2]
    summary.append((filt, k, f"{med:.3f}", len(vals)))

with open('results/baseline_seq_summary.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filter', 'k', 'median_compute_s', 'runs'])
    for row in summary:
        writer.writerow(row)

print("Updated results/baseline_seq_summary.csv:")
for row in summary:
    print(row)

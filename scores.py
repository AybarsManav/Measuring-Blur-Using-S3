from scipy.stats import pearsonr, spearmanr
import os

#fill in manually 
sharpness_scores = []
mos_values = []


mos_path = 'tid2008\mos_with_names.txt'  
mos_scores = {}

with open(mos_path, 'r') as f:
    for line in f:
        parts = line.strip().split()
        filename = parts[0]
        score = float(parts[1])
        mos_scores[filename] = score


image_folder = 'distorted_images'

for filename in os.listdir(image_folder):
    if filename.endswith('.bmp'):
        path = os.path.join(image_folder, filename)

        if filename in mos_scores:

            mos_values.append(mos_scores[filename])
plcc, _ = pearsonr(sharpness_scores, mos_values)
srcc, _ = spearmanr(sharpness_scores, mos_values)

print(f"PLCC (Pearson): {plcc:.4f}")
print(f"SRCC (Spearman): {srcc:.4f}")
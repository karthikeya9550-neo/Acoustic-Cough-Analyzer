
import glob
import os

folder = "../SPRSound-main/BioCAS2022/train2022_wav"

files = glob.glob(os.path.join(folder, "*.wav"))

print("Training audio files found:", len(files))
print("\nFirst 5 files:")

for file in files[:5]:
    print(os.path.basename(file))

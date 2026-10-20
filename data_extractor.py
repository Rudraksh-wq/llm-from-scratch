
import os
import random
import glob
import pyarrow.parquet as pq
from tqdm import tqdm

folder_path = r"D:\download"

output_train = r"D:\download\output_train.txt"
output_val = r"D:\download\output_val.txt"
vocab_file = r"D:\download\vocab.txt"

sample_rate = 0.01
train_ratio = 0.90
random.seed(42)

files = glob.glob(os.path.join(folder_path, "*.parquet"))

if not files:
    raise FileNotFoundError(
        f"No Parquet files found in {folder_path}. "
        "Check whether your files are inside a subfolder."
    )

files.sort()
print("Parquet files found:", len(files))

train_chars = set()
val_chars = set()

train_count = 0
val_count = 0

with open(output_train, "w", encoding="utf-8") as train_file, \
     open(output_val, "w", encoding="utf-8") as val_file:

    for file_path in tqdm(files, desc="Reading files"):
        parquet_file = pq.ParquetFile(file_path)

        for batch in parquet_file.iter_batches(
            batch_size=1000,
            columns=["text"]
        ):
            texts = batch.column("text").to_pylist()

            for text in texts:
                if random.random() >= sample_rate:
                    continue

                if not isinstance(text, str) or not text.strip():
                    continue

                if random.random() < train_ratio:
                    train_file.write(text + "\n")
                    train_chars.update(text)
                    train_count += 1
                else:
                    val_file.write(text + "\n")
                    val_chars.update(text)
                    val_count += 1

vocab = train_chars | val_chars

with open(vocab_file, "w", encoding="utf-8") as f:
    for char in sorted(vocab):
        f.write(char + "\n")

print("\nProcessing completed!")
print("Training documents:", train_count)
print("Validation documents:", val_count)
print("Vocabulary size:", len(vocab))
print("Training output:", output_train)
print("Validation output:", output_val)
print("Vocabulary output:", vocab_file)

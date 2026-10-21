import os
import glob
import random
import pyarrow.parquet as pq
from tqdm import tqdm

folder_path = r"D:\download"

output_train = os.path.join(folder_path, "output_train.txt")
output_val = os.path.join(folder_path, "output_val.txt")
vocab_file = os.path.join(folder_path, "vocab.txt")

sample_rate = 0.01
train_ratio = 0.90
batch_size = 5000
buffer_size = 1000
seed = 42

random.seed(seed)

for path in [output_train, output_val, vocab_file]:
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)

files = sorted(glob.glob(os.path.join(folder_path, "**", "*.parquet"), recursive=True))
if not files:
    files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))

if not files:
    raise FileNotFoundError(f"No Parquet files found in {folder_path}")

print("Parquet files found:", len(files))

vocab = set()
train_count = 0
val_count = 0

train_buffer = []
val_buffer = []

with open(output_train, "w", encoding="utf-8", buffering=1024 * 1024) as train_file, \
     open(output_val, "w", encoding="utf-8", buffering=1024 * 1024) as val_file:

    for file_path in tqdm(files, desc="Reading files"):
        try:
            parquet_file = pq.ParquetFile(file_path)
        except Exception:
            print(f"Skipping unreadable file: {file_path}")
            continue

        column_name = "text" if "text" in parquet_file.schema.names else parquet_file.schema.names[0]

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=[column_name]):
            texts = batch.column(column_name).to_pylist()

            for text in texts:
                if sample_rate < 1.0 and random.random() >= sample_rate:
                    continue

                if not isinstance(text, str):
                    continue

                text = text.replace("\r", "").strip()
                if not text:
                    continue

                vocab.update(text)

                if random.random() < train_ratio:
                    train_buffer.append(text)
                    train_count += 1
                    if len(train_buffer) >= buffer_size:
                        train_file.write("\n".join(train_buffer) + "\n")
                        train_buffer.clear()
                else:
                    val_buffer.append(text)
                    val_count += 1
                    if len(val_buffer) >= buffer_size:
                        val_file.write("\n".join(val_buffer) + "\n")
                        val_buffer.clear()

    if train_buffer:
        train_file.write("\n".join(train_buffer) + "\n")
        train_buffer.clear()

    if val_buffer:
        val_file.write("\n".join(val_buffer) + "\n")
        val_buffer.clear()

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

from datasets import load_dataset
CUAD_DATASET_NAME = "theatticusproject/cuad"

#dataset = load_dataset(CUAD_DATASET_NAME)
dataset = load_dataset(CUAD_DATASET_NAME,download_mode="force_redownload")
train = dataset["train"]
print(f"Dataset loaded: {dataset}")


for record in dataset.
   doc={}
   doc.text = record.context
   doc.title = record.title



# View first example
print(train[0])

# View specific fields
print(train[0]["title"])
print(train[0]["paragraphs"])


# # Iterate
# for example in train:
#     print(example["title"])
#     print(example["paragraphs"])

# # Pandas if you prefer
# df = train.to_pandas()
# df.head()


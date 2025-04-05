from datasets import load_dataset

# 加载数据集
# dataset = load_dataset("/home/lmh/.cache/huggingface/hub/datasets--timm--resisc45")
dataset = load_dataset("/home/lmh/.cache/huggingface/hub/datasets--timm--resisc45")
# 查看数据集信息
print(dataset)

# 访问训练集
train_data = dataset["train"]

# 查看第一条数据
print(train_data[0])
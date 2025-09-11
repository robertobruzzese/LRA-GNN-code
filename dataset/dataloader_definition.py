# dataloader_definition.py
from torch.utils.data import DataLoader
from dataset.embedding_dataset import EmbeddingDataset

embedding_dataset = EmbeddingDataset("embeddings/train")
embedding_loader = DataLoader(embedding_dataset, batch_size=1, shuffle=True)

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Tuple

# Configuración
TOKENIZER = get_tokenizer("basic_english")
MAX_VOCAB_SIZE = 25_000
PAD_IDX = 0
UNK_IDX = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Usando dispositivo:", device)
if device.type == 'cuda':
    print("Nombre de la GPU:", torch.cuda.get_device_name(0))
    
# Función para leer las reseñas
def read_data(root_dir: str) -> Tuple[str, str]:
    """Leer reseñas desde archivos de texto en las carpetas pos/neg"""
    samples = []
    for label in ['pos', 'neg']:
        folder = os.path.join(root_dir, label)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            with open(path, encoding='utf-8') as f:
                text = f.read()
            samples.append((label, text))
    return samples

# Dataset personalizado para IMDB
class IMDBDataset(Dataset):
    def __init__(self, root_dir: str, vocab, tokenizer):
        self.samples = read_data(root_dir)
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        label, text = self.samples[idx]
        tokens = self.tokenizer(text)
        indices = [self.vocab[token] for token in tokens]
        label = 1 if label == "pos" else 0  # Convertir a 1 para positivo y 0 para negativo
        return torch.tensor(indices, dtype=torch.long), label

# Función para construir el vocabulario
def build_vocab_from_data(samples):
    vocab = build_vocab_from_iterator(
        (TOKENIZER(text) for _, text in samples),
        max_tokens=MAX_VOCAB_SIZE,
        specials=["<pad>", "<unk>"]
    )
    vocab.set_default_index(UNK_IDX)
    return vocab

# Función de procesamiento de batch
def collate_batch(batch, vocab):
    text_list, label_list, lengths = [], [], []
    for text, label in batch:
        text_list.append(text)
        label_list.append(label)
        lengths.append(len(text))
    
    padded = pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    lengths = torch.tensor(lengths, dtype=torch.long)
    labels = torch.tensor(label_list, dtype=torch.float)
    return padded, lengths, labels.unsqueeze(1)

# Función que prepara los DataLoaders
def get_data_loaders(batch_size: int = 64, root_dir: str = None) -> Tuple[DataLoader, DataLoader, DataLoader, torch.nn.Module, int]:
    # Usar ruta absoluta basada en la ubicación de este archivo
    if root_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        root_dir = os.path.join(base_dir, "aclImdb")
    # Leer los datos
    train_samples = read_data(os.path.join(root_dir, 'train'))
    test_samples = read_data(os.path.join(root_dir, 'test'))

    # Construir vocabulario
    vocab = build_vocab_from_data(train_samples)

    # Crear Dataset y DataLoader
    train_dataset = IMDBDataset(os.path.join(root_dir, 'train'), vocab, TOKENIZER)
    test_dataset = IMDBDataset(os.path.join(root_dir, 'test'), vocab, TOKENIZER)

    # Dividir el entrenamiento en train y validación
    split = int(0.8 * len(train_dataset))
    train_data, valid_data = torch.utils.data.random_split(train_dataset, [split, len(train_dataset) - split])

    # Crear DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_batch(x, vocab))
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_batch(x, vocab))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_batch(x, vocab))

    return train_loader, valid_loader, test_loader, vocab, PAD_IDX

# Test básico (opcional, solo para verificar)
if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab, pad_idx = get_data_loaders(batch_size=4)
    for x, lengths, y in train_loader:
        print(f"Batch x shape: {x.shape}")
        print(f"Lengths: {lengths}")
        print(f"Labels: {y.view(-1)}")
        break

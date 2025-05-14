import torch
import torch.nn as nn
from src.data_utils import get_data_loaders
from src.model import SentimentLSTM


def binary_accuracy(preds, y):
    rounded = torch.round(torch.sigmoid(preds))
    correct = (rounded == y).float()
    return correct.sum() / len(correct)


if __name__ == "__main__":
    # Configuraciones
    BATCH_SIZE = 64
    MODEL_PATH = "model.pt"

    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoaders
    train_loader, valid_loader, test_loader, vocab, pad_idx = get_data_loaders(batch_size=BATCH_SIZE)

    # Cargar modelo
    model = SentimentLSTM(vocab_size=len(vocab), pad_idx=pad_idx)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # Función de pérdida
    criterion = nn.BCEWithLogitsLoss()

    # Evaluar en test set
    test_loss, test_acc = 0, 0
    with torch.no_grad():
        for x, lengths, y in test_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            predictions = model(x, lengths)
            test_loss += criterion(predictions, y).item()
            test_acc += binary_accuracy(predictions, y).item()

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    print(f"Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}")

import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.data_utils import get_data_loaders
from src.model import SentimentLSTM


def binary_accuracy(preds, y):
    """
    Retorna precisión binaria: porcentaje de predicciones correctas
    """
    # Aplicar sigmoide y redondear
    rounded = torch.round(torch.sigmoid(preds))
    correct = (rounded == y).float()
    return correct.sum() / len(correct)


def train_model(
    model,
    train_loader,
    valid_loader,
    optimizer,
    criterion,
    device,
    num_epochs=5,
    save_path="model.pt"
):
    best_valid_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss, epoch_train_acc = 0, 0
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad()
            predictions = model(x, lengths)
            loss = criterion(predictions, y)
            acc = binary_accuracy(predictions, y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_acc += acc.item()

        valid_loss, valid_acc = evaluate(model, valid_loader, criterion, device)

        print(f"Epoch {epoch+1}:",
              f"Train Loss: {epoch_train_loss/len(train_loader):.3f},",
              f"Train Acc: {epoch_train_acc/len(train_loader):.2f},",
              f"Val. Loss: {valid_loss:.3f},",
              f"Val. Acc: {valid_acc:.2f}")

        # Guardar el mejor modelo
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)


def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss, epoch_acc = 0, 0
    with torch.no_grad():
        for x, lengths, y in iterator:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            predictions = model(x, lengths)
            loss = criterion(predictions, y)
            acc = binary_accuracy(predictions, y)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


if __name__ == "__main__":
    # Configuraciones
    BATCH_SIZE = 64
    NUM_EPOCHS = 5
    LEARNING_RATE = 1e-3
    SAVE_PATH = "model.pt"

    # Dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DataLoaders
    train_loader, valid_loader, test_loader, vocab, pad_idx = get_data_loaders(batch_size=BATCH_SIZE)

    # Modelo
    model = SentimentLSTM(vocab_size=len(vocab), pad_idx=pad_idx)
    model = model.to(device)

    # Optimizador y función de pérdida
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(device)

    # Entrenar
    train_model(model, train_loader, valid_loader, optimizer, criterion, device, NUM_EPOCHS, SAVE_PATH)
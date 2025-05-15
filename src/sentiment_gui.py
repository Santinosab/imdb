import sys
import os
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton
from data_utils import get_data_loaders, TOKENIZER, PAD_IDX
from model import SentimentLSTM

class SentimentApp(QWidget):
    def __init__(self, model, vocab, device):
        super().__init__()
        self.model = model
        self.vocab = vocab
        self.device = device
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Clasificador de Sentimiento IMDB')
        self.layout = QVBoxLayout()
        self.label = QLabel('Ingresa una reseña de película:')
        self.text_edit = QTextEdit()
        self.button = QPushButton('Predecir Sentimiento')
        self.result = QLabel('')
        self.button.clicked.connect(self.predict)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.text_edit)
        self.layout.addWidget(self.button)
        self.layout.addWidget(self.result)
        self.setLayout(self.layout)

    def predict(self):
        text = self.text_edit.toPlainText()
        sentiment, prob = predict_sentiment(text, self.model, self.vocab, TOKENIZER, PAD_IDX, self.device)
        self.result.setText(f"Predicción: {sentiment} (prob: {prob:.2f})")

def predict_sentiment(text, model, vocab, tokenizer, pad_idx, device, max_len=500):
    model.eval()
    tokens = tokenizer(text)
    indices = [vocab[token] for token in tokens][:max_len]
    tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    length = torch.tensor([len(indices)]).cpu()
    with torch.no_grad():
        output = model(tensor, length)
        prob = torch.sigmoid(output)
        pred = int(prob >= 0.5)
    return ("Positiva" if pred == 1 else "Negativa"), float(prob)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, _, vocab, _ = get_data_loaders(batch_size=4)
    model = SentimentLSTM(vocab_size=len(vocab), pad_idx=PAD_IDX)
    # Ruta absoluta al modelo, siempre desde la raíz del proyecto
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    app = QApplication(sys.argv)
    window = SentimentApp(model, vocab, device)
    window.show()
    sys.exit(app.exec_())

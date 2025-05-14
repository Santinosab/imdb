import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=100, hidden_size=128, output_size=1, pad_idx=0):
        super(SentimentLSTM, self).__init__()
        
        # Capa de Embedding para convertir las palabras en vectores
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_idx)
        
        # Capa LSTM: recibe la secuencia de embeddings y procesa la información secuencial
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, bidirectional=True)
        
        # Capa de Dropout para regularización
        self.dropout = nn.Dropout(0.5)
        
        # Capa lineal para predecir la salida (sentimiento: positivo o negativo)
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 debido a la bidireccionalidad

    def forward(self, x, lengths):
        # x tiene forma (batch_size, seq_len)
        # lengths es la longitud de cada secuencia en el batch

        # Embedding
        x = self.embedding(x)
        
        # LSTM: la salida es (batch_size, seq_len, hidden_size * 2) ya que es bidireccional
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Extraemos la salida final de la secuencia (la última celda oculta de la LSTM)
        # Concatenamos la última celda oculta de la LSTM hacia adelante y hacia atrás
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)  # Dim=1 para concatenar hacia adelante y hacia atrás
        
        # Aplicamos Dropout
        hidden_cat = self.dropout(hidden_cat)
        
        # Salida
        output = self.fc(hidden_cat)
        
        return output

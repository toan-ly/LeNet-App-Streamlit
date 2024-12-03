import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator, Vocab
from torchtext import vocab

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, kernel_sizes, num_classes):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.num_classes = num_classes
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, 
                      out_channels=n_filters, 
                      kernel_size=k,
                      stride=1
            ) for k in kernel_sizes])
            
        self.fc = nn.Linear(len(kernel_sizes) * n_filters, num_classes)

    def forward(self, text):
        batch_size, seq_len = text.shape
        output = self.embedding(text.T).transpose(1, 2)

        output = [F.relu(conv(output)) for conv in self.convs]
        output = [F.max_pool1d(x, x.size(-1)).squeeze(dim=-1) for x in output]
        output = torch.cat(output, dim=1)
        output = self.fc(output)
        return output
    
def load_model(model_path, vocab_size=10000, embedding_dim=100, num_classes=2):
    model = TextCNN(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        kernel_sizes=[3, 4, 5],
        n_filters=100,
        num_classes=num_classes
    )
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    return model

tokenizer = get_tokenizer('basic_english')
idx2label = {0: 'Negative', 1: 'Positive'}
model = load_model('apps/text_cnn_app/checkpoints/text_cnn_model.pt')
vocabulary = torch.load('apps/text_cnn_app/checkpoints/vocabulary.pth')

def inference(sentence, vocabulary, model):
    encoded_sentence = vocabulary(tokenizer(sentence))
    encoded_sentence = torch.tensor(encoded_sentence)
    encoded_sentence = torch.unsqueeze(encoded_sentence, 1)

    with torch.no_grad():
        predictions = model(encoded_sentence)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return round(p_max.item(), 2) * 100, yhat.item()

def main():
    st.title('Sentiment Analysis')
    st.title('Model: Text CNN. Dataset: NTC-SCV')
    text_input = st.text_input("Sentence: ", "Đồ ăn ở quán này quá tệ luôn!")
    p, idx = inference(text_input, vocabulary, model)
    label = idx2label[idx]
    st.success(f'Prediction: {label}, Probability: ({p:.2f}%)')

if __name__ == '__main__':
    main()     
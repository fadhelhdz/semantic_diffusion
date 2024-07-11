import torch
from transformers import DistilBertTokenizer, DistilBertModel

class BERT:
    def __init__(self, device):
        # Load the BERT model configuration and model
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.model = model.to(device)
        

    def get_embedding(self, device, sentence):
        """Get the BERT embedding for a sentence."""
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            output = self.model(**inputs)
        # Extract the [CLS] embedding
        return output.last_hidden_state[:, 0, :]

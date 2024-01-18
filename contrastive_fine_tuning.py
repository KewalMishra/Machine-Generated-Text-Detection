import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer
import tqdm
# Load data
train_df = pd.read_csv("en_train.csv")
test_df = pd.read_csv("en_test.csv")
device = 'cuda:1'
# Load tokenizer and BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased').to(device)

# Define model
class ContrastiveBERT(nn.Module):
    def __init__(self, bert):
        super(ContrastiveBERT, self).__init__()
        self.bert = bert
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        outputs1 = self.bert(input_ids1, attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids2, attention_mask=attention_mask2)
        embeddings1 = outputs1[1]
        embeddings2 = outputs2[1]
        x = torch.abs(embeddings1 - embeddings2)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return x

model = ContrastiveBERT(bert).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

# Tokenize data
train_tokens1 = tokenizer.batch_encode_plus(train_df['source'].tolist(),
                                            padding=True,
                                            truncation=True,
                                            max_length=128)
train_tokens2 = tokenizer.batch_encode_plus(train_df['source'].tolist(),
                                            padding=True,
                                            truncation=True,
                                            max_length=128)
test_tokens1 = tokenizer.batch_encode_plus(test_df['source'].tolist(),
                                           padding=True,
                                           truncation=True,
                                           max_length=128)
test_tokens2 = tokenizer.batch_encode_plus(test_df['source'].tolist(),
                                           padding=True,
                                           truncation=True,
                                           max_length=128)

# Convert tokens to tensors
train_inputs1 = torch.tensor(train_tokens1['input_ids']).to(device)
train_masks1 = torch.tensor(train_tokens1['attention_mask']).to(device)
train_inputs2 = torch.tensor(train_tokens2['input_ids']).to(device)
train_masks2 = torch.tensor(train_tokens2['attention_mask']).to(device)
train_labels = torch.tensor(train_df['label'].tolist()).to(device)
test_inputs1 = torch.tensor(test_tokens1['input_ids']).to(device)
test_masks1 = torch.tensor(test_tokens1['attention_mask']).to(device)
test_inputs2 = torch.tensor(test_tokens2['input_ids']).to(device)
test_masks2 = torch.tensor(test_tokens2['attention_mask']).to(device)
test_labels = torch.tensor(test_df['label'].tolist()).to(device)

# Train model
epochs = 1
batch_size = 6
for epoch in tqdm.tqdm(range(epochs)):
    for i in range(0, len(train_inputs1), batch_size):
        inputs1 = train_inputs1[i:i+batch_size].to(device)
        masks1 = train_masks1[i:i+batch_size].to(device)
        inputs2 = train_inputs2[i:i+batch_size].to(device)
        masks2 = train_masks2[i:i+batch_size].to(device)
        labels = train_labels[i:i+batch_size].to(device)
        optimizer.zero_grad()
        outputs = model(inputs1, masks1, inputs2, masks2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate model
model.eval()
with torch.no_grad():
    outputs = model(test_inputs1, test_masks1, test_inputs2, test_masks2)
    predictions = torch.argmax(outputs, dim=1)
    accuracy = torch.sum(predictions == test_labels) / len(test_labels)
    print("Accuracy:", accuracy)

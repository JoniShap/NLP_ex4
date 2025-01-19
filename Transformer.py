from torch import nn

import data_loader
from exercise_blanks import DataManager, TRAIN, VAL, TEST, ONEHOT_AVERAGE


def train_transformer_for_sentiment():
    """
    Train and evaluate a transformer model (distilroberta-base) for sentiment analysis.
    Using learning rate 1e-5, weight decay = 0, and batch size from previous sections.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import Dataset, DataLoader
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    class SentimentDataset(Dataset):
        def __init__(self, sentences, labels, tokenizer, max_length=128):
            self.encodings = tokenizer([' '.join(s.text) for s in sentences],
                                       truncation=True,
                                       padding=True,
                                       max_length=max_length)
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            return item

        def __len__(self):
            return len(self.labels)

    def train_epoch(model, data_loader, optimizer, criterion, device):
        model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(data_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get logits and compute loss
            loss = criterion(outputs.logits.squeeze(), labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Calculate accuracy
            predictions = (torch.sigmoid(outputs.logits.squeeze()) > 0.5).float()
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions / total_predictions
        return avg_loss, accuracy

    def evaluate(model, data_loader, criterion, device):
        model.eval()
        total_loss = 0.0
        predictions_list = []
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits.squeeze(), labels)

                total_loss += loss.item()

                predictions = (torch.sigmoid(outputs.logits.squeeze()) > 0.5).float()
                predictions_list.extend(predictions.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        accuracy = sum(1 for x, y in zip(predictions_list, labels_list) if x == y) / len(labels_list)
        avg_loss = total_loss / len(data_loader)
        return avg_loss, accuracy

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create data manager
    data_manager = DataManager(data_type=ONEHOT_AVERAGE, batch_size=64)

    # Load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=1).to(device)
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')

    # Create datasets
    train_dataset = SentimentDataset(data_manager.sentences[TRAIN],
                                     [sent.sentiment_class for sent in data_manager.sentences[TRAIN]],
                                     tokenizer)
    val_dataset = SentimentDataset(data_manager.sentences[VAL],
                                   [sent.sentiment_class for sent in data_manager.sentences[VAL]],
                                   tokenizer)
    test_dataset = SentimentDataset(data_manager.sentences[TEST],
                                    [sent.sentiment_class for sent in data_manager.sentences[TEST]],
                                    tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    test_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Training
    n_epochs = 2
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}/{n_epochs}")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # Final test set evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Add special subsets evaluation here:
    test_sentences = data_manager.sentences[TEST]
    print(f"\nTotal number of test sentences: {len(test_sentences)}")

    # Get indices for special subsets
    negated_indices = data_loader.get_negated_polarity_examples(test_sentences)
    rare_indices = data_loader.get_rare_words_examples(test_sentences, data_manager.sentiment_dataset, num_sentences=60)

    print(f"Number of negated polarity examples: {len(negated_indices)}")
    print(f"Number of rare word examples: {len(rare_indices)}")

    # Get all predictions on test set
    model.eval()
    all_predictions = []
    all_labels = []

    print("\nGathering all test predictions...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']  # Keep labels on CPU

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.sigmoid(outputs.logits.squeeze()).cpu()  # Move predictions to CPU

            all_predictions.extend(predictions.numpy())
            all_labels.extend(labels.numpy())

    # Convert to tensors
    test_predictions_tensor = torch.tensor(all_predictions)
    test_labels_tensor = torch.tensor(all_labels)

    # Calculate accuracies manually for special subsets
    negated_predictions = test_predictions_tensor[negated_indices]
    negated_labels = test_labels_tensor[negated_indices]
    rare_predictions = test_predictions_tensor[rare_indices]
    rare_labels = test_labels_tensor[rare_indices]

    # Manual accuracy calculation
    negated_correct = ((negated_predictions >= 0.5).float() == negated_labels).float().mean()
    rare_correct = ((rare_predictions >= 0.5).float() == rare_labels).float().mean()

    # Debug information
    print("\nDetailed accuracy analysis:")
    print(f"Main test accuracy: {test_acc:.3f}")
    print(f"Negated examples accuracy: {negated_correct:.3f}")
    print(f"Rare words accuracy: {rare_correct:.3f}")

    print("\nRare words prediction details:")
    for i in range(min(10, len(rare_indices))):
        idx = rare_indices[i]
        pred = test_predictions_tensor[idx]
        label = test_labels_tensor[idx]
        binary_pred = 1 if pred >= 0.5 else 0
        correct = binary_pred == label
        print(f"Example {i}:")
        print(f"  Sentence: {' '.join(test_sentences[idx].text)}")
        print(f"  Raw prediction: {pred:.3f}")
        print(f"  Binary prediction: {binary_pred}")
        print(f"  True label: {label}")
        print(f"  Correct: {correct}")

    # Distribution of predictions
    print("\nPrediction distribution:")
    print(f"Rare predictions mean: {rare_predictions.mean():.3f}")
    binary_rare_preds = (rare_predictions >= 0.5).float()
    print(f"Rare predictions distribution:")
    print(f"Zeros: {(binary_rare_preds == 0).sum().item()}")
    print(f"Ones: {(binary_rare_preds == 1).sum().item()}")

    # [Rest of the code remains the same - plotting and return statement]
    plt.figure(figsize=(12, 4))
    # ... [plotting code remains unchanged]

    return {
        'test_accuracy': test_acc,
        'negated_accuracy': negated_correct.item(),
        'rare_accuracy': rare_correct.item(),
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs
    }
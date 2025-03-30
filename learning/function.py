import torch
import tqdm


def train_single_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for waveform, label in tqdm.tqdm(dataloader, desc="Training Batches", leave=False, miniters=10):
        waveform, label = waveform.to(device), label.to(device)

        # loss와 prediction 계산
        logits = model(waveform)
        loss = criterion(logits, label)
        # loss = criterion(logits.float(), label.float().view(-1, 1))

        # 역전파 및 gradients 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * waveform.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Training loss: {epoch_loss:.4f}')


def train(model, dataloader, criterion, optimizer, device, epochs):
    for epoch in tqdm.tqdm(range(epochs), desc="Training epochs", total=epochs):
        print(f'\nepoch: {epoch+1}/{epochs}')
        train_single_epoch(model, dataloader, criterion, optimizer, device)
        print('-----------------------------------------------------------------')

        # 검증
        if epoch % 100 == 0:
            valid(model, dataloader, criterion, device)

    valid(model, dataloader, criterion, device)
    print('Finished Training')


def valid(model, dataloader, criterion, device):
    model.eval()
    valid_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for waveform, label in tqdm.tqdm(dataloader, desc="Validation batches", leave=False, miniters=10):
            waveform, label = waveform.to(device), label.to(device)
            logits = model(waveform)
            loss = criterion(logits, label)
            valid_loss += loss.item() * waveform.size(0)

            # 예측값 계산
            _, preds = torch.max(logits, 1)
            correct += torch.sum(preds == label).item()
            total += label.size(0)

        valid_total_loss = valid_loss / len(dataloader.dataset)
        accuracy = correct / total if total > 0 else 0
        print(f'Validation loss: {valid_total_loss:.4f}, Accuracy: {accuracy:.4f}')
        print('-----------------------------------------------------------------')
from tqdm import tqdm
import torch


def train(
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        num_epochs: int,
        device: torch.device,
):
    model.to(device)

    history = {
        'train': {
            'total': 0,
            'loss': [],
            'accuracy': [],
            'output_pred': [],
            'output_true': []
        },
        'valid': {
            'total': 0,
            'loss': [],
            'accuracy': [],
            'output_pred': [],
            'output_true': []
        }
    }

    for epoch in range(1, num_epochs + 1):

        # đưa model vào trạng thái train
        model.train()

        train_loss = 0.0
        train_steps = 0
        train_total = 0
        train_correct = 0

        train_output_pred = []
        train_output_true = []

        print(f'Epoch {epoch}/{num_epochs}:')
        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.squeeze(1)

            # zero the parameter gradients
            for opt in optimizer:
                opt.zero_grad()

            # Passing the batch down the model
            outputs = model(inputs)

            # forward + backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()

            # For every possible optimizer performs the gradient update
            for opt in optimizer:
                opt.step()

            train_loss += loss.cpu().item()
            train_steps += 1

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            train_output_pred += outputs.argmax(1).cpu().tolist()
            train_output_true += labels.tolist()

        # đưa model vào trạng thái đánh giá (evaluation)
        model.eval()

        val_loss = 0.0
        val_steps = 0
        val_total = 0
        val_correct = 0

        val_output_pred = []
        val_output_true = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.squeeze(1)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)  # labels size [N, 1] trong đó N là số sample, cũng là size của mini-batch
                val_correct += (predicted == labels).sum().item()

                val_output_pred += outputs.argmax(1).cpu().tolist()
                val_output_true += labels.tolist()

        history['train']['total'] = train_total
        history['train']['loss'].append(train_loss / train_steps)
        history['train']['accuracy'].append(train_correct / train_total)
        history['train']['output_pred'] = train_output_pred
        history['train']['output_true'] = train_output_true

        history['valid']['total'] = val_total
        history['valid']['loss'].append(val_loss / val_steps)
        history['valid']['accuracy'].append(val_correct / val_total)
        history['valid']['output_pred'] = val_output_pred
        history['valid']['output_true'] = val_output_true

        print(
            f'loss: {train_loss / train_steps} - acc: {train_correct / train_total} - val_loss: {val_loss / val_steps} - val_acc: {val_correct / val_total}')

    print("Finished training")

    return history

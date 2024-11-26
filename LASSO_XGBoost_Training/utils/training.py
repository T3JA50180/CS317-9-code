from .imports import *
scaler = GradScaler('cuda')

def train_step(device,
               model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn,
               optimizer: torch.optim.Optimizer):
    train_loss, train_acc = 0, 0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        with autocast(device_type='cuda'):
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

        train_loss += loss
        train_acc += accuracy_fn(y_pred, y)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc


def test_step(device,
              model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy_fn):
    test_loss, test_acc = 0, 0
    model.eval()

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            with autocast(device_type='cuda'):
                y_pred = model(X)
                loss = loss_fn(y_pred, y)

            test_loss += loss
            test_acc += accuracy_fn(y_pred, y)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)

    return test_loss, test_acc


def train(device,
          model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module(),
          accuracy_fn,
          epochs: int,
          model_name,
          scheduler=None,):
    
    best_acc = 0.0

    results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    for epoch in tqdm(range(epochs)):
        
        start_time = timer()
        train_loss, train_acc = train_step(device,
                                           model,
                                           train_dataloader,
                                           loss_fn,
                                           accuracy_fn,
                                           optimizer,)
          
        test_loss, test_acc = test_step(device,
                                        model,
                                        test_dataloader,
                                        loss_fn,
                                        accuracy_fn,)
        end_time = timer()

        print(f'Epoch: {epoch:2d} | '
              f'Train_Loss: {train_loss:.4f} | ',
              f'Train_Acc: {train_acc:.4f} | ',
              f'Test Loss: {test_loss:.4f} | ',
              f'Test Acc: {test_acc:.4f} | ',
              f'Time: {end_time - start_time:.4f} seconds')

        results['train_loss'].append(train_loss.item())
        results['train_acc'].append(train_acc.item())
        results['test_loss'].append(test_loss.item())
        results['test_acc'].append(test_acc.item())
        
        if test_acc.item() > best_acc:
            best_acc = test_acc.item()
            torch.save(model, f'model_saves/{model_name}.pth')
            
        scheduler.step()

    return results
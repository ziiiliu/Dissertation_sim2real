import torch
import torch.optim as optim

def train_differential(model, X_train, y_train, X_val, y_val, 
        epochs=1000, model_save_path="ckpt/best_simplepredictor.pt",
        lr=1e-4, opt='adam', writer=None):
    if opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Optimizer choice unknown")
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    loss_func = torch.nn.MSELoss()

    losses = []

    for epoch in range(epochs):
    
        best_val_loss = float('inf')
        
        prediction = model(X_train)
        # print(X_train[0], y_train[0], prediction[0])
        loss = loss_func(prediction, y_train)

        optimizer.zero_grad()
        loss.backward()         
        optimizer.step()
    #     scheduler.step()
        if epoch % 10 == 0:
            writer.add_scalar('train_loss', loss, global_step=epoch)
        print(f'epoch number: {epoch+1}, MSE Loss: {loss.data}')
        
        if epoch % 100 == 0:
            val_y_preds = model(X_val)
            val_loss = loss_func(val_y_preds, y_val)
            writer.add_scalar('validation_loss', val_loss, global_step=epoch)
            print('Validation Loss: ', val_loss.data)
            losses.append(val_loss.data.item())
            if val_loss.data < best_val_loss:
                best_val_loss = val_loss.data
                torch.save(model.state_dict(), model_save_path)


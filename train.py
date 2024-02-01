from Data.prepare_dataloader import get_from_loader
from Models.UNet_Architecture import Build_UNet
import torch 
from Losses.calculate_loss import DiceLoss
import torch.optim as optim 
from Metrics.Get_Metrics import calculate_metrics
from tqdm import tqdm 
from operator import add
from Plots.plot_figures import plot_Accuracy, plot_IOU_Jaccard, plot_loss
import matplotlib.pyplot as plt
import yaml
import json
import os 

with open('./config/train_config.yaml', 'r') as config_file:
    config_params = yaml.safe_load(config_file)
    model_config = json.dumps(config_params)

def training_phase(train_dataloader, test_dataloader):
    overall_train_loss_per_epoch = []
    overall_test_loss_per_epoch = []
    overall_train_jaccard_per_epoch = []
    overall_test_jaccard_per_epoch = []
    overall_train_acc_per_epoch = []
    overall_test_acc_per_epoch = []

    num_epochs = 500
    model = Build_UNet(num_classes=4).to(device)
    loss_function = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    checkpoint_path = 'model.pth'
    start_epoch = 1
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")


    for epoch in range(1,num_epochs+1):
        epoch_train_loss = 0.0
        epoch_test_loss = 0.0  
        metrics_score = [0.0, 0.0]

        model.train()
        for batch_data in tqdm(train_dataloader, desc=f'Training Epoch {epoch}/{num_epochs}', unit='epoch'):
            inputs = batch_data[0].float().to(device)
    #         print(inputs.shape)
            labels = batch_data[1].float().to(device)
    #         print(labels.shape)

    #         print(torch.max(inputs))
    #         print(torch.min(inputs))

    #         print(torch.max(labels))
    #         print(torch.min(labels))

            optimizer.zero_grad()
            outputs = model(inputs)

    #         print(torch.max(outputs))
    #         print(torch.min(outputs))
    #         print(outputs.shape)
    #         print(labels.shape)

            train_loss = loss_function(outputs, labels)
            # train_loss.requires_grad = True
            train_loss.backward()
    #         print(outputs)
    #         print('------------------')
    #         print(labels)
            score = calculate_metrics(outputs, labels)
            metrics_score = list(map(add, metrics_score, score))

            optimizer.step()
    #         print(train_loss.item())
    #         print(type(epoch_train_loss))
            epoch_train_loss += train_loss.item()

            epoch_train_loss = epoch_train_loss/len(train_dataloader)
            epoch_train_jaccard = metrics_score[0]/len(train_dataloader)
            epoch_train_acc = metrics_score[1]/len(train_dataloader)

        overall_train_loss_per_epoch.append(train_loss.item())
        overall_train_jaccard_per_epoch.append(epoch_train_jaccard)
        overall_train_acc_per_epoch.append(epoch_train_acc)
            
            
        model.eval()
        metrics_score = [0.0, 0.0]
        with torch.no_grad():
            for batch_data in tqdm(test_dataloader, desc=f'Test Epoch {epoch}/{num_epochs}', unit='epoch'):
                inputs = batch_data[0].float().to(device)
                labels = batch_data[1].float().to(device)
                outputs = model(inputs)

                test_loss = loss_function(outputs, labels)

                score = calculate_metrics(outputs, labels)
                metrics_score = list(map(add, metrics_score, score))

                optimizer.step()
    #             print(test_loss.item())
                epoch_test_loss += test_loss.item()
                epoch_test_loss = epoch_train_loss/len(test_dataloader)
                epoch_test_jaccard = metrics_score[0]/len(test_dataloader)
                epoch_test_acc = metrics_score[1]/len(test_dataloader)
                
    #             print(epoch_test_loss)
                
            overall_test_loss_per_epoch.append(test_loss.item()) 
            overall_test_jaccard_per_epoch.append(epoch_test_jaccard)
            overall_test_acc_per_epoch.append(epoch_test_acc)


        print(f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {epoch_train_loss:.4f}, '
            f'Train Jaccard: {epoch_train_jaccard:.4f}, '
            f'Train Accuracy: {epoch_train_acc:.4f}, '
            f'Test Loss: {epoch_test_loss:.4f}, '
            f'Test Jaccard: {epoch_test_jaccard:.4f}, '
            f'Test Accuracy: {epoch_test_acc:.4f}, ')
        
        
    return model,num_epochs,optimizer, train_loss, overall_train_loss_per_epoch, overall_train_jaccard_per_epoch, overall_train_acc_per_epoch, overall_test_loss_per_epoch, overall_test_jaccard_per_epoch, overall_test_acc_per_epoch

    



if __name__ == '__main__':

    # print(config_params["training_params"]["loss_function"])
    train_batch_size = config_params["training_params"]["train_batch_size"]
    val_batch_size = config_params["training_params"]["val_batch_size"]
    learning_rate = config_params["training_params"]["learning_rate"]
    num_classes = config_params["training_params"]["num_classes"]
    loss_function = config_params["training_params"]["loss_function"]
    device_type = config_params["training_params"]["device_type"]
    optimizer = config_params["training_params"]["optimizer"]

    # device_name = config_params[]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    # t2_location = '/mnt/Enterprise2/shirshak/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t2.nii'
    # t1ce_location = '/mnt/Enterprise2/shirshak/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1ce.nii'
    # flair_location = '/mnt/Enterprise2/shirshak/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*flair.nii'
    # mask_location = '/mnt/Enterprise2/shirshak/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*seg.nii'

    # train_dataloader, test_dataloader = get_from_loader(t2_location,t1ce_location,flair_location,mask_location)


    model, num_epochs,optimizer, loss, overall_train_loss_per_epoch, overall_train_jaccard_per_epoch, overall_train_acc_per_epoch, overall_test_loss_per_epoch, overall_test_jaccard_per_epoch, overall_test_acc_per_epoch = training_phase(train_dataloader,test_dataloader)

    plot_loss(num_epochs, overall_train_loss_per_epoch, overall_test_loss_per_epoch)
    plot_IOU_Jaccard(num_epochs, overall_train_jaccard_per_epoch, overall_test_jaccard_per_epoch)
    plot_Accuracy(num_epochs, overall_train_acc_per_epoch, overall_test_acc_per_epoch)

    for batch_data in test_dataloader:
        inputs = batch_data[0].float().to(device)
        labels = batch_data[1].float().to(device)
        outputs = model(inputs)

    # model.eval()
    with torch.no_grad():
        x = X_test[0].to(device, dtype=torch.float32)
        y = y_test[0].to(device, dtype=torch.float32)
        x = x.unsqueeze(0)
    #     x= x.transpose(1, 4).transpose(1, 2).transpose(2, 3)
        y_pred = model(x)
    #     print(y_pred.squeeze(0,1).shape)
        x = torch.squeeze(x).transpose(0, 3).transpose(0,1).transpose(1, 2)
        y_pred = torch.squeeze(y_pred).transpose(0, 3).transpose(0,1).transpose(1, 2)
        print(f'X : {x.shape}')
        y = y.transpose(0, 3).transpose(0,1).transpose(1, 2)
        y = torch.argmax(y,dim=3)
        print(f'y : {y.shape}')
        # if dim =4 then it will be carried on the 4th axis which is the argmax containing of the 4 different classes 
        y_pred = torch.argmax(y_pred, dim = 3) # columnwise --> dim=0 #rowwise --> dim=1 
        print(f'y_pred : {y_pred.shape}')


    n_slice = 55
    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(x[:,:,n_slice,1].cpu(), cmap='gray')
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(y[:,:,n_slice].cpu())
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(y_pred[:,:, n_slice].cpu())
    plt.show()




    # Save the model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, 'saved_model.pth')

    # Load the model
    loaded_model = Build_UNet()
    checkpoint = torch.load('saved_model.pth')
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']


















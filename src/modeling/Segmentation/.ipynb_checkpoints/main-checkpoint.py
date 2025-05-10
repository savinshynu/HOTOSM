import torch
import pandas as pd
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import UNet
from dataset import HotosmDataset

def main(save_model=False, model_save_path="./"):
    learning_rate = 3e-4
    batch_size = 8
    epochs = 10 # no. of epochs
    # Location of the images and mask
    data_path = "/Users/savin/Omdena-Projects/HOTOSM/data/Mask/road_mask.npz"

    # define the device  here
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device}")

    # calling dataset class
    # This is an effective and clean way to access data based on Pytorch
    # DataLoader class will utilize this custom dataset object to split data into batches, shuffle and 
    # effective retrieval
    train_dataset = HotosmDataset(data_path)
    generator = torch.Generator().manual_seed(42) # setting up a manual seed for consistent results
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    # calling the dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)

    # call the model
    # Ideally you want to load all the model and data to the device

    model = UNet(in_channels=3, num_classes=4).to(device) # defining the model
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate) # Adam version suited for transformers and UNET architectures.
    criterion = nn.CrossEntropyLoss() # still a question

    train_losses = []
    val_losses = []

    for e in range(epochs): # starting the training iterations
        model.train() # setting the model to train mode
        train_running_loss = 0 # calculate loss for each epoch
        print(f"calculating loss across {len(train_dataloader)} batches")

        for img, mask in tqdm(train_dataloader): # data contains the image and mask

            #print(f"Processing epoch:{e}, batch {num}")
            img = img.to(device) # each of this should be [batch_size, channels, height, width]
            
            mask = mask.to(device)

            #print(img.shape, mask.shape)

            ypred = model(img) # This basically outputs the logits and not probabilities. 
            # output shape:  [batch, classes, height, width]
            # mask shape: [batch, height, width] == where pixel value is the index of each class
            # A computational graph is created each during the feed forward network and remeber all the calculations
            # Because CrossEntropyLoss internally does:

            # 1. calculate p = -LogSoftmax(logits) for each pixel
            # 2. loss of single pixel = p[class_index] or selecting -log(softmax(logits)) for the true class
            # 4. Averaging over all the pixels of the image and the batch.
            # 5. Basically generalize the normal multi-class classification along different pixels in an image.

            optimizer.zero_grad() # Zero the gradients in each batch, otherwise they will accumulate

            loss = criterion(ypred, mask) # ypred[nbatch, nclass, height, width], mask:[nbatch, height, width] pixel = 0,1,2...N classes
            train_running_loss += loss.item()

            loss.backward() # Back propogation step, calculate the gradient of each parameter and updates the parameter.grad field. 
            # This calculate gradient needs to be cleared after each batch, other wise it will start accumulating.
            optimizer.step() # update the parameters using the calculated gradients in the .grad field of each parameter

            # Delete unnecessary thing to free up GPU memory
            del img, mask, ypred, loss

            # Empty unused cache
            torch.mps.empty_cache()

       
        train_loss = train_running_loss/len(train_dataloader) # Average across batches

        train_losses.append(train_loss)

        # Now set the model to the evaluation mode
        model.eval()
        val_running_loss = 0
        with torch.no_grad(): # stop the backpropogation for the evaluation mode and no need to create the compuation graph in the forward run
            for img, mask in tqdm(val_dataloader): # data contains the image and mask
                img = img.to(device)
                mask = mask.to(device)

                ypred = model(img)
                loss = criterion(ypred, mask)

                val_running_loss += loss.item()

                del img, mask, ypred

                # Empty unused cache
                torch.mps.empty_cache()
       
            
            val_loss = val_running_loss/len(val_dataloader)
            val_losses.append(val_loss)
        # print statistics across for each epoch
        print("*"*20)
        print(f"Train loss, epoch {e} : {train_loss:0.4f}")
        print(f"Validation loss, epoch {e} : {val_loss:0.4f}")
        print("*"*20)
    

    if save_model:
        # save the history as a data frame
        df = pd.DataFrame(data={"train_loss":train_losses, "val_loss":val_losses})
        df.to_csv(model_save_path+"history")

        #save pytorch model
        #torch.save(model.state_dict(),  model_save_path)
        # save to torchscript format, no model defining is needed
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(model_save_path+'unet_roads.pt') # Save


if __name__ == "__main__":
    main(save_model=True, model_save_path="models/")
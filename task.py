import torch
import torch.nn as nn
import torch.optim as optim
import time
import random
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from MixUp import MixUp
from torchvision.transforms.functional import to_pil_image
from PIL import ImageDraw
import argparse
from datetime import datetime
from torchvision.models import vision_transformer
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging



# Set seed for reproducibility
def set_seed(seed_value):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(15)

# Create a directory to save the trained models is it does not exist
if not os.path.exists('trained_models'):
    os.makedirs('trained_models')

# Create a directory to save the results is it does not exist
if not os.path.exists('results'):
    os.makedirs('results')

# Create a directory to save the output of Mixup class demo is it does not exist
if not os.path.exists('mixup'):
    os.makedirs('mixup')

# Create a directory to save logs
if not os.path.exists('logs'):
    os.makedirs('logs')

# Create a custom logger
logger = logging.getLogger(__name__)

# Set the level of this logger. Only messages of this severity level or above will be logged.
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler(f'logs/task3_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

# Create a console handler
console_handler = logging.StreamHandler()

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class MyModel(nn.Module):
    def __init__(self, model, num_classes, device):
        super(MyModel, self).__init__()
        if model == 'Vision Transformer':

            self.model = vision_transformer.vit_b_32(weights='ViT_B_32_Weights.DEFAULT')

            self.model.dropout = 0.2


            # Get the number of input features of the last layer
            num_in_features = self.model.heads.head.in_features
            # Replace the last layer with a new one
            self.model.heads.head = nn.Linear(num_in_features, num_classes)

            for param in self.model.parameters():
                param.requires_grad = False

            for param in self.model.encoder.layers.encoder_layer_6.parameters():
                param.requires_grad = True
            for param in self.model.encoder.layers.encoder_layer_7.parameters():
                param.requires_grad = True
            for param in self.model.encoder.layers.encoder_layer_8.parameters():
                param.requires_grad = True
            for param in self.model.encoder.layers.encoder_layer_9.parameters():
                param.requires_grad = True
            for param in self.model.encoder.layers.encoder_layer_10.parameters():
                param.requires_grad = True
            for param in self.model.encoder.layers.encoder_layer_11.parameters():
                param.requires_grad = True
            for param in self.model.encoder.ln.parameters():
                param.requires_grad = True
            for param in self.model.heads.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

def train(model, trainloader, valiloader, criterion, optimizer, scheduler, sampling_method, alpha, device, max_epochs=10):
    logger.info("_"*20 + "Training" + "_"*20)
    logger.info(f'Vision Transformer with MixUp network with sampling method {sampling_method}')
    # logger.info the hyperparameters
    logger.info(f'EPOCHS: {max_epochs}, loss function: {type(criterion)}, optimizer: {type(optimizer)},Batch Size: {trainloader.batch_size}, Dropout: {model.model.dropout}')
    logger.info('\n\n')
    logger.info('Using Pretrained Weights as Initialization. Freezing the weights of initial layers and training the last few layers of the model.')
    logger.info('\n\n')
    prev_lr = optimizer.param_groups[0]['lr']

    training_results = {'epoch': [], 'training_loss': [], 'validation_loss': [], 'validation_accuracy': [], 'validation_precision': [], 'validation_f1_score': [], 'learning_rate': [], 'time_per_epoch': []}
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=10)
            inputs = inputs.to(device)
            one_hot_labels = 1.0*one_hot_labels.to(device) #1.0 to ensure to use right datatype
            Mixer = MixUp(alpha, sampling_method)
            inputs, one_hot_labels = Mixer.mix(inputs, one_hot_labels)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, one_hot_labels)
            loss = criterion(outputs, one_hot_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


        # Validation
        model.eval()
        accuracy = 0.0
        val_loss = 0.0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for _, val_data in enumerate(valiloader, 0):
                val_images, val_labels = val_data
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                # Forward pass
                val_outputs = model(val_images)

                # Calculate validation loss
                one_hot_val_labels = torch.nn.functional.one_hot(val_labels, num_classes=10)
                one_hot_val_labels = 1.0 * one_hot_val_labels.to(device)
                val_loss += criterion(val_outputs, one_hot_val_labels)

                # Calculate accuracy
                val_predictions = torch.argmax(val_outputs, 1, keepdim=False)
                accuracy += torch.sum(val_predictions == val_labels)

                # Collect labels and predictions for precision and F1-score calculation
                all_labels.extend(val_labels.view(-1).cpu().numpy())
                all_predictions.extend(val_predictions.view(-1).cpu().numpy())

        # Calculate average validation loss and accuracy
        val_loss /= len(valiloader)
        accuracy = 100.0 * accuracy.item() / len(valiloader.dataset)

        # Convert lists to numpy arrays
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)

        # Calculate true positives, false positives, and false negatives
        true_positive = np.sum((all_predictions == 1) & (all_labels == 1))
        false_positive = np.sum((all_predictions == 1) & (all_labels == 0))
        false_negative = np.sum((all_predictions == 0) & (all_labels == 1))

        # Calculate precision and recall
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)

        # Calculate F1-score
        f1_score = 2 * (precision * recall) / (precision + recall)

        end_time = time.time()
        time_per_epoch = end_time - start_time

        scheduler.step(val_loss)

        if prev_lr != optimizer.param_groups[0]['lr']:
            logger.info(f'Learning rate changed from {prev_lr} to {optimizer.param_groups[0]["lr"]} on plateau of validation loss.')
            prev_lr = optimizer.param_groups[0]['lr']

        logger.info(f'Epoch {epoch + 1:2}/{max_epochs:2}, Training Loss: {running_loss / len(trainloader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%, Validation Precision: {precision:.4f}, Validation F1-score: {f1_score:.4f}, Learning Rate: {scheduler.get_last_lr()[0]}, Time per epoch: {time_per_epoch:.2f} s')

        # Save results
        training_results['epoch'].append(epoch + 1)
        training_results['training_loss'].append(running_loss / len(trainloader))
        training_results['validation_loss'].append(val_loss)
        training_results['validation_accuracy'].append(accuracy)
        training_results['validation_precision'].append(precision)
        training_results['validation_f1_score'].append(f1_score)
        training_results['learning_rate'].append(scheduler.get_last_lr()[0])
        training_results['time_per_epoch'].append(time_per_epoch)


    logger.info('Training done.')
    logger.info('Total time elapsed: {:.2f} s'.format(sum(training_results['time_per_epoch'])))

    # Save trained model
    logger.info(f'Saving trained model as vision_transformer_with_mixup_sampling_method_{sampling_method}.pth in trained_models directory.')
    torch.save(model.state_dict(), f'trained_models/vision_transformer_with_mixup_sampling_method_{sampling_method}.pth')

    # Save training results
    logger.info(f'Saving training results as training_results_with_mixup_sampling_method_{sampling_method}.csv in results directory.')
    with open(f'results/training_results_with_mixup_sampling_method_{sampling_method}.csv', 'w') as file:
        file.write('epoch,training_loss,validation_loss,validation_accuracy,validation_precision,validation_f1_score,learning_rate,time_per_epoch\n')
        for i in range(max_epochs):
            file.write(f"{training_results['epoch'][i]},{training_results['training_loss'][i]},{training_results['validation_loss'][i]},{training_results['validation_accuracy'][i]},{training_results['validation_precision'][i]},{training_results['validation_f1_score'][i]},{training_results['learning_rate'][i]},{training_results['time_per_epoch'][i]}\n")

    logger.info('Model saved.')

def test(model, testloader,criterion, device,sampling_method=1):
    logger.info('_'*20 + 'Testing' + '_'*20)
    model.eval()
    accuracy = 0.0
    test_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for _, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            # Forward pass
            test_outputs = model(test_images)

            # Calculate test loss
            one_hot_test_labels = torch.nn.functional.one_hot(test_labels, num_classes=10)
            one_hot_test_labels = 1.0 * one_hot_test_labels.to(device)
            test_loss += criterion(test_outputs, one_hot_test_labels)

            # Calculate accuracy
            val_predictions = torch.argmax(test_outputs, 1, keepdim=False)
            accuracy += torch.sum(val_predictions == test_labels)

            # Collect labels and predictions for precision and F1-score calculation
            all_labels.extend(test_labels.view(-1).cpu().numpy())
            all_predictions.extend(val_predictions.view(-1).cpu().numpy())

    # Calculate average training loss and accuracy
    test_loss /= len(testloader)
    accuracy = 100.0 * accuracy.item() / len(testloader.dataset)

    # Convert lists to numpy arrays
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate true positives, false positives, and false negatives
    true_positive = np.sum((all_predictions == 1) & (all_labels == 1))
    false_positive = np.sum((all_predictions == 1) & (all_labels == 0))
    false_negative = np.sum((all_predictions == 0) & (all_labels == 1))

    # Calculate precision and recall
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    # Calculate F1-score
    f1_score = 2 * (precision * recall) / (precision + recall)

    logger.info(f'Test Results with sampling method {sampling_method}:')
    logger.info(f'Test Loss: {test_loss:.4f}')
    logger.info(f'Test Accuracy: {accuracy:.2f}%')
    logger.info(f'Test Precision: {precision:.4f}')
    logger.info(f'Test F1-score: {f1_score:.4f}')


    return test_loss, accuracy, precision, f1_score


def visualize_results(model, testloader, criterion, device, num_images,sampling_method=1):

    model.eval()
    counter = 0
    montage_image = Image.new('RGB', (192*6, 192*6))

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    results = test(model, testloader, criterion, device,sampling_method)
    draw_montage = ImageDraw.Draw(montage_image)
    draw_montage.text((10,10), f'Test Accuracy: {results[0]:.2f}%', fill=(255,255,255))

    with torch.no_grad():
        for _, test_data in enumerate(testloader, 0):
            test_images, test_labels = test_data
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)
            test_outputs = model(test_images)
            test_predictions = torch.argmax(test_outputs, 1, keepdim=False)
            for i in range(test_images.size(0)):
                counter += 1
                img = Image.fromarray(((test_images[i].cpu().numpy().transpose((1, 2, 0)) + 1) * 0.5 * 255).astype(np.uint8))
                # img=Image.fromarray(torch.cat(test_images[i].cpu().detach().numpy().split(1, 0), 3).squeeze() / 2 * 255 + 0.5 * 255).permute(1, 2,0).numpy().astype('uint8')
                # img = img.resize((img.width//2 , img.height//2))
                d = ImageDraw.Draw(img)
                # If the prediction is correct, write the class name in green, otherwise write it in red
                if test_predictions[i].item() == test_labels[i].item():
                    d.text((10,10), f'Ground truth: {classes[test_labels[i].item()]}\nPrediction: {classes[test_predictions[i].item()]}', fill=(0,255,0))
                else:
                    d.text((10,10), f'Ground truth: {classes[test_labels[i].item()]}\nPrediction: {classes[test_predictions[i].item()]}', fill=(255,0,0))
                # d.text((10,10), f'Ground truth: {classes[test_labels[i].item()]}\nPrediction: {classes[test_predictions[i].item()]}', fill=(255,255,255))
                montage_image.paste(img, ((counter-1)%6*192, (counter-1)//6*192))
                if counter == num_images:
                    montage_image.save(f'results/result_with_sampling_method_{sampling_method}.png')
                    logger.info(f'Results saved successfully as results/result_with_sampling_method_{sampling_method}.png')
                    return results

# initialize main
def train_main(trainloader, valiloader, device, MAX_EPOCHS, LEARNING_RATE):


    # Create MixUp instances
    mixer_1 = MixUp(alpha=0.4, sampling_method=1)
    mixer_2 = MixUp(alpha=0.4, sampling_method=2)


    # Create model
    model = MyModel(model='Vision Transformer', num_classes=10, device=device)
    model.to(device)

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    train(model, trainloader, valiloader, criterion, optimizer, scheduler, 1, 0.4, device, MAX_EPOCHS)

    #zero the weights of the model
    model = MyModel(model='Vision Transformer', num_classes=10, device=device)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    train(model, trainloader, valiloader, criterion, optimizer, scheduler, 2, 0.4, device, MAX_EPOCHS)




def main(MODE=1):
    logger.info('\n\n')
    logger.info("#" * 20 + " Start " + "#" * 20)
    logger.info('\n\n')

    if torch.cuda.is_available():
        # Get device properties
        device = torch.device(f'cuda:{0}')
        properties = torch.cuda.get_device_properties(device)

        # logger.info device properties
        logger.info(f'Device Name: {properties.name}')
        logger.info(f'Total Memory: {properties.total_memory / (1024 ** 3):.2f} GB')

        # logger.info memory currently allocated and memory reserved
        logger.info(f'Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 ** 3):.2f} GB')
        logger.info(f'Memory Reserved: {torch.cuda.memory_reserved(device) / (1024 ** 3):.2f} GB')
    else:
        logger.info('CUDA is not available. Make sure you have installed the necessary drivers. Set Device to CPU. ')

    # Set batch size
    BATCH_SIZE = 256
    MAX_EPOCHS = 20
    LEARNING_RATE = 5e-5

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomRotation(10),  # Random rotation between -10 and 10 degrees
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # Random tilt (shear) and scale
        transforms.GaussianBlur(5),  # Apply Gaussian blur with kernel size 5
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.GaussianBlur(5),  # Apply Gaussian blur with kernel size 5
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    logger.info('Loading datasets...')
    # Load datasets
    developmentset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Get the length of the development set
    length_dev = len(developmentset)

    # Calculate the lengths of the training and validation sets
    length_train = int(0.9 * length_dev)
    length_val = length_dev - length_train

    # Randomly split the development set
    trainset, valset = torch.utils.data.random_split(developmentset, [length_train, length_val])

    # Apply data augmentation to the training set
    trainset.dataset.transform = transform_train

    logger.info('Applying data augmentation and gaussian transformations to the training set...')

    logger.info("Datasets loaded successfully.")

    # Create data loaders
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valiloader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    if MODE==1 or MODE==0:
        logger.info('\n\n')
        logger.info("#"*20 + " MixUp class demo " + "#"*20)
        logger.info('\n\n')
        # Get the first 16 images and labels
        images, labels = next(iter(testloader))
        images_subset = images[:16].to(device)
        labels_subset = labels[:16].to(device)

        # Create MixUp instances
        mixer_1 = MixUp(alpha=0.4, sampling_method=1)
        mixer_2 = MixUp(alpha=0.4, sampling_method=2)

        logger.info('Mixing images using MixUp class demo...')
        # Mix images and labels
        images_mixed_1, labels_mixed_1 = mixer_1.mix(images_subset, labels_subset)
        images_mixed_2, labels_mixed_2 = mixer_2.mix(images_subset, labels_subset)

        # Save mixed images
        for i, images_mixed in enumerate([images_mixed_1, images_mixed_2], start=1):
            image = to_pil_image(torch.cat(images_mixed.split(1, 0), 3).squeeze() / 2 + 0.5)
            with open(f"mixup/mixup_with_sampling_method_{i}.png", 'wb') as f:
                image.save(f)
                logger.info(f"MixUp with sampling method {i} saved successfully as mixup/mixup_with_sampling_method_{i}.png")
        logger.info('MixUp class demo completed successfully.')

    if MODE==2 or MODE==0:
        logger.info('\n\n')
        logger.info("#"*20 + " Training block Initiated " + "#"*20)
        logger.info('\n\n')
        train_main(trainloader, valiloader, device, MAX_EPOCHS, LEARNING_RATE)

    if MODE==3 or MODE==0:
        logger.info('\n\n')
        logger.info("#"*20 + " Testing and Visualization block Initiated " + "#"*20)
        logger.info('\n\n')
        # load the trained models
        logger.info('Loading trained models...')
        model1 = MyModel(model='Vision Transformer', num_classes=10, device=device)
        model1.load_state_dict(torch.load('trained_models/vision_transformer_with_mixup_sampling_method_1.pth'))
        model1.to(device)
        model1.eval()
        logger.info('Model 1 (sampling method 1) loaded successfully.')

        model2 = MyModel(model='Vision Transformer', num_classes=10, device=device)
        model2.load_state_dict(torch.load('trained_models/vision_transformer_with_mixup_sampling_method_2.pth'))
        model2.to(device)
        model2.eval()
        logger.info('Model 2 (sampling method 2) loaded successfully.')
        logger.info('\n\n')
        logger.info('Visualizing results...\n')
        criterion = torch.nn.CrossEntropyLoss()
        results1 = visualize_results(model1, testloader,criterion, device, 36, sampling_method=1)
        results2 = visualize_results(model2, testloader,criterion, device, 36, sampling_method=2)

        # Compare the results of both the models
        logger.info('-'*20 + 'Results' + '-'*20)

        logger.info('Training Summary of model with sampling method 1:\n')

        print_flag = False

        val_dict_1={'epoch': [], 'training_loss': [], 'validation_loss': [], 'validation_accuracy': [], 'validation_precision': [], 'validation_f1_score': [], 'learning_rate': [], 'time_per_epoch': []}
        # Load the results of model with sampling method 1 and logger.info the results in tabular format
        with open('results/training_results_with_mixup_sampling_method_1.csv', 'r') as file:
            lines = file.readlines()
            headers = lines[0].strip().split(',')
            if print_flag:
                logger.info("-" * 190)
                logger.info("| {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} |".format(*headers))
                logger.info("-" * 190)
            for line in lines[1:]:
                values = line.strip().split(',')
                if print_flag:
                    logger.info("| {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} |".format(*values))
                    logger.info("-" * 190)
                val_dict_1['epoch'].append(int(values[0]))
                val_dict_1['training_loss'].append(float(values[1]))
                val_dict_1['validation_loss'].append(float(values[2]))
                val_dict_1['validation_accuracy'].append(float(values[3]))
                val_dict_1['validation_precision'].append(float(values[4]))
                val_dict_1['validation_f1_score'].append(float(values[5]))
                val_dict_1['learning_rate'].append(float(values[6]))
                val_dict_1['time_per_epoch'].append(float(values[7]))

        logger.info('\n\n')
        logger.info('Training Summary of model with sampling method 2:\n')

        val_dict_2={'epoch': [], 'training_loss': [], 'validation_loss': [], 'validation_accuracy': [], 'validation_precision': [], 'validation_f1_score': [], 'learning_rate': [], 'time_per_epoch': []}
        # Load the results of model with sampling method 2 and logger.info the results in tabular format
        with open('results/training_results_with_mixup_sampling_method_2.csv', 'r') as file:
            lines = file.readlines()
            headers = lines[0].strip().split(',')
            if print_flag:
                logger.info("-" * 190)
                logger.info("| {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} |".format(*headers))
                logger.info("-" * 190)
            for line in lines[1:]:
                values = line.strip().split(',')
                if print_flag:
                    logger.info("| {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} |".format(*values))
                    logger.info("-" * 190)
                val_dict_2['epoch'].append(int(values[0]))
                val_dict_2['training_loss'].append(float(values[1]))
                val_dict_2['validation_loss'].append(float(values[2]))
                val_dict_2['validation_accuracy'].append(float(values[3]))
                val_dict_2['validation_precision'].append(float(values[4]))
                val_dict_2['validation_f1_score'].append(float(values[5]))
                val_dict_2['learning_rate'].append(float(values[6]))
                val_dict_2['time_per_epoch'].append(float(values[7]))

        logger.info('\n\n')
        # logger.info('Testing Summary:\n')
        # # Compare the results of both the models
        # # Define the table headers
        # headers = ["Method", "Test Accuracy", "Test Precision", "Test F1-score", "Validation Accuracy", "Validation Precision", "Validation F1-score", "Total Training Time"]
        #
        # # Define the data
        # data = [
        #     ["Sampling Method 1", results1[0], results1[1], results1[2], val_dict_1['validation_accuracy'][-1], val_dict_1['validation_precision'][-1], val_dict_1['validation_f1_score'][-1], sum(val_dict_1['time_per_epoch'])],
        #     ["Sampling Method 2", results2[0], results2[1], results2[2], val_dict_2['validation_accuracy'][-1], val_dict_2['validation_precision'][-1], val_dict_2['validation_f1_score'][-1], sum(val_dict_2['time_per_epoch'])]
        # ]
        # # logger.info the table headers
        # logger.info("-" * 190)
        # logger.info("| {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} |".format(*headers))
        # logger.info("-" * 190)
        # # logger.info the data
        # for values in data:
        #     logger.info("| {:<20} | {:<20.2f} | {:<20.4f} | {:<20.4f} | {:<20.2f} | {:<20.4f} | {:<20.4f} | {:<20.2f} |".format(*values))
        #     logger.info("-" * 190)

        logger.info('-'*20 + 'Summary' + '-'*20)


        # summary of loss values, speed, metric on training and validation
        logger.info('\nTraining Summary:\n')
        headers= ["Method", "Training Loss", "Validation Loss", "Validation Accuracy", "Validation Precision", "Validation F1-score", "Total Training Time(s)"]

        data = [
            ["Sampling Method 1", val_dict_1['training_loss'][-1], val_dict_1['validation_loss'][-1], val_dict_1['validation_accuracy'][-1], val_dict_1['validation_precision'][-1], val_dict_1['validation_f1_score'][-1], sum(val_dict_1['time_per_epoch'])],
            ["Sampling Method 2", val_dict_2['training_loss'][-1], val_dict_2['validation_loss'][-1], val_dict_2['validation_accuracy'][-1], val_dict_2['validation_precision'][-1], val_dict_2['validation_f1_score'][-1], sum(val_dict_2['time_per_epoch'])]
        ]
        
        logger.info("-" * 170)
        logger.info("| {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<25} |".format(*headers))
        logger.info("-" * 170)
        for values in data:
            logger.info("| {:<20} | {:<20.4f} | {:<20.4f} | {:<20.2f} | {:<20.4f} | {:<20.4f} | {:<25.2f} |".format(*values))
            logger.info("-" * 170)
            
        # summary of loss values and the metrics on the holdout test set. Comparing the results
        # with those obtained during development.

        logger.info('\nTesting Summary:\n')
        # Compare the results of both the models
        # Define the table headers
        headers = ["Method", "Test Loss(CrossEntropy)", "Vali Loss(CrossEntropy)", "Test Accuracy", "Validation Accuracy", "Test Precision", "Validation Precision", "Test F1-score", "Validation F1-score"]

        # Define the data
        data = [
            ["Sampling Method 1", results1[0], val_dict_1['validation_loss'][-1], results1[1], val_dict_1['validation_accuracy'][-1], results1[2], val_dict_1['validation_precision'][-1], results1[3], val_dict_1['validation_f1_score'][-1]],
            ["Sampling Method 2", results2[0], val_dict_2['validation_loss'][-1], results2[1], val_dict_2['validation_accuracy'][-1], results2[2], val_dict_2['validation_precision'][-1], results2[3], val_dict_2['validation_f1_score'][-1]]
        ]
        # logger.info the table headers
        logger.info("-" * 220)
        logger.info("| {:<20} | {:<25} | {:<25} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | ".format(*headers))
        logger.info("-" * 220)
        # logger.info the data
        for values in data:
            logger.info("| {:<20} | {:<25.4f} | {:<25.4f} | {:<20.2f} | {:<20.2f} | {:<20.4f} | {:<20.4f} | {:<20.4f} | {:<20.4f} |".format(*values))
            logger.info("-" * 220)

        logger.info('\n\n')
        logger.info("#"*20 + " End " + "#"*20)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train and test the Vision Transformer model with MixUp.')
    parser.add_argument('--mode', type=int, default=0, help='Mode: 0 for All, 1 for MixupDemo, 2 for Training, 3 for Testing and Visualization. Default: 0.')
    args = parser.parse_args()
    MODE = args.mode
    assert MODE in [0, 1, 2, 3], "Mode must be 0, 1, 2 or 3."

    if MODE == 0:
        logger.info('Mode: All')
    elif MODE == 1:
        logger.info('Mode: MixupDemo')
    elif MODE == 2:
        logger.info('Mode: Training')
    elif MODE == 3:
        logger.info('Mode: Testing and Visualization')
    main(MODE)

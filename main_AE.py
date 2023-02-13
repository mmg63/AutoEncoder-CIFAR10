import os
from autoencoder_Classes import Autoencoder, Encoder, Decoder
from dataset_manipulate import *
import matplotlib.pyplot as plt

from utilities import compare_imgs, find_similar_images, visualize_reconstructions
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np

parser = argparse.ArgumentParser("main_AE")
parser.add_argument('--dataset', default='CIFAR10')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-03)
parser.add_argument('--weight-decay', type=float, default=1e-3)
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--save_model', action='store_true', default=True)


# def plot_images(samples_batch, x_hat_batch, dataset_classes_idx, labels, number_of_columns_for_plot: int = 4):
def plot_images(samples_batch, x_hat_batch, dataset_classes_idx, labels, idx: int = 4):
    # cordinate plot windows for plotting images
    # batch_size = samples_batch.shape[0]
    # number_of_row = batch_size * 2 // number_of_columns_for_plot
    # if (batch_size % number_of_columns_for_plot) > 0:
    #     number_of_row += 1
    # number_of_row *= 2
    # fig = plt.figure()
    with torch.no_grad():
    #     for i in range(batch_size - 1):
    #         fig.add_subplot(number_of_row, number_of_columns_for_plot, i + 1)
    #         plt.imshow(torch.clip(samples_batch[i].movedim(0, -1), 0, 1))
    #         plt.axis('off')
    #         plt.title(dataset_classes_idx[labels[i]])
    #         fig.add_subplot(number_of_row, number_of_columns_for_plot, i + 2)
    #         plt.imshow(torch.clip(x_hat_batch[i].movedim(0, -1), 0, 1))
    #         plt.axis('off')
    #         plt.title(dataset_classes_idx[labels[i]] + ' model')
    #     plt.show()
        plt.subplot(3, 2, 1)
        dataset_classes[labels[0]]
        plt.imshow(torch.clip(samples_batch[idx].movedim(0, -1), 0, 1))
        plt.title(dataset_classes_idx[labels[idx]] + ' original')

        plt.subplot(3, 2, 2)
        plt.imshow(torch.clip(x_hat_batch[idx].movedim(0, -1), 0, 1))
        plt.title(dataset_classes_idx[labels[idx]] + ' trough model')

        plt.subplot(3, 2, 3)
        plt.imshow(torch.clip(samples_batch[idx + 1].movedim(0, -1), 0, 1))
        plt.title(dataset_classes_idx[labels[idx+1]] + ' original')

        plt.subplot(3, 2, 4)
        plt.imshow(torch.clip(x_hat_batch[idx + 1].movedim(0, -1), 0, 1))
        plt.title(dataset_classes_idx[labels[idx+1]] + ' trough model')

        plt.subplot(3, 2, 5)
        plt.imshow(torch.clip(samples_batch[idx + 2].movedim(0, -1), 0, 1))
        plt.title(dataset_classes_idx[labels[idx+2]] + ' original')

        plt.subplot(3, 2, 6)
        plt.imshow(torch.clip(x_hat_batch[idx + 2].movedim(0, -1), 0, 1))
        plt.title(dataset_classes_idx[labels[idx+2]] + ' trough model')

        plt.show()

if __name__ == "__main__":
    # ============
    # parse args and path
    # ============
    args = parser.parse_args()

    absolute_path = os.path.dirname(os.path.realpath(__file__)) + '/AE-ckps/'
    file_name = "AE_parameters"
    # ============
    # load dataset
    # ============
    train_loader = load_train_loader(data_dir='../CIFAR10', batch_size=args.batch_size,
                                     download_allowed=True, pin_memory=False)
    test_loader = load_test_loader('../CIFAR10', batch_size=args.batch_size, shuffle=True, pin_memory=False)

    dataset_classes = CIFAR10_classes()

    # ============
    # phase: define model and optimizer
    # ============
    model_AE = Autoencoder(base_channel_size=32, latent_dim=384)
    optim_AE = optim.Adam(model_AE.parameters(), lr=args.lr,)  # weight_decay=args.weight_decay)
    print(model_AE)

    if args.load_model:
    # ============
    # load model
    # ============
        checkpoint = torch.load(absolute_path + file_name + ".ckp")
        model_AE.load_state_dict(checkpoint['model_AE_state_dict'])
        optim_AE.load_state_dict(checkpoint['optimizer_AE'])
        init_epoch = checkpoint['epoch'] + 1
        # history = (np.load(absolute_path + file_name + '-history.npy', model_history_acc)).tolist()
        os.system('say the model is loaded successfully')
        print('Model is loaded')

        # plot some images
        itr = iter(train_loader)
        samples, labels = itr.next()
        latent, x_hat = model_AE(samples)
        plot_images(samples, x_hat, dataset_classes, labels, 4)

    else:
        os.system('say Training phase')
        # model_history_acc = []
        # ============
        # phase: Train
        # ============
        for epoch in range(args.epochs):
            running_loss = 0.0
            model_acc = 0.0
            total_images = 0

            for i, data in enumerate(train_loader, 0):

                samples, labels = data
                optim_AE.zero_grad()
                latent_space_features, x_hat = model_AE(samples)
                train_loss = F.mse_loss(x_hat, samples, reduction='none')
                train_loss = train_loss.sum(dim=[1, 2, 3]).mean(dim=[0])
                # print('AE training Loss = {}'.format(train_loss.data))
                train_loss.backward()
                optim_AE.step()

                # _, predicted = torch.max(latent_space_features.data, 1)
                # total_images += labels.size(0)
                # num_correct = (predicted == labels).sum().item()
                # model_acc += num_correct / (len(train_loader) * samples.size(0))
                # running_loss += train_loss.item() / len(train_loader)

            print("Epoch: ", epoch, " - model_loss: " + str(running_loss))  # + ' - train_acc: ' + str(model_acc))

            # ============
            # phase: test accuracy calculation
            # ============
            # with torch.no_grad():
            #     total_correct = 0.0
            #     total_images = 0.0
            #     test_acc = 0.0
            #     for data in test_loader:
            #         images, labels = torch.autograd.Variable(data[0]), torch.autograd.Variable(data[1])
            #         latent_space_features, x_hat = model_AE(images)
            #         _, predicted = torch.max(latent_space_features.data, 1)
            #         total_images += labels.size(0)
            #         test_acc += (predicted == labels).sum().item() / (len(test_loader) * images.size(0))
            #     print('Model Test accuracy:', test_acc)
            #
            # speech_sound_text = 'say The model ' + \
            #                     'has reached to the train accuracy level of ' + \
            #                     str(model_acc * 100) + '%, and test accuracy level of ' + \
            #                     str(test_acc * 100) + '%, at epoch of ' + str(epoch)
            # os.system(speech_sound_text)
            # plot_images(samples, x_hat, dataset_classes, labels)

            # os.system('say see your output as plot')
            # plot_images(samples, x_hat, dataset_classes, labels, 1)

            print('Saving model at epoch {}'.format(epoch))
            # ============
            # Saving the model, optimizer, and accuracies.
            # ============
            # model_history_acc .append([model_acc, test_acc])
            checkpoint = {'epoch': epoch,
                          'model_AE_state_dict': model_AE.state_dict(),
                          'optimizer_AE': optim_AE.state_dict()}
            torch.save(checkpoint, absolute_path + file_name + '.ckp')
            torch.save(model_AE, absolute_path + file_name + '.mdl')  # saving entire model
        # np.save(absolute_path + file_name + '-history.npy', model_history_acc)

    # ============
    # phase: plot arbitrary images in training model
    # ============

import torch
import torch.nn.functional as F
from autoencoder_Classes import *
import torchvision
import pytorch_lightning as pl
import matplotlib.pyplot as plt


class GenerateCallback(pl.Callback):

    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs # Images to reconstruct during training
        self.every_n_epochs = every_n_epochs # Only save those images every N epochs (otherwise tensorboard gets quite large)

    def on_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            input_imgs = self.input_imgs.to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs = pl_module(input_imgs)
                pl_module.train()
            # Plot and add to tensorboard
            imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
            grid = torchvision.utils.make_grid(imgs, nrow=2, normalize=True, range=(-1,1))
            trainer.logger.experiment.add_image("Reconstructions", grid, global_step=trainer.global_step)


def compare_imgs(img1, img2, title_prefix=""):
    # Calculate MSE loss between both images
    loss = F.mse_loss(img1, img2, reduction="sum")
    # Plot images for visual comparison
    grid = torchvision.utils.make_grid(torch.stack([img1, img2], dim=0), nrow=2, normalize=True, range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(4,2))
    plt.title("%s Loss: %4.2f" % (title_prefix, loss.item()))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


# for i in range(2):
#     # Load example image
#     img, _ = train_dataset[i]
#     img_mean = img.mean(dim=[1,2], keepdims=True)
#
#     # Shift image by one pixel
#     SHIFT = 1
#     img_shifted = torch.roll(img, shifts=SHIFT, dims=1)
#     img_shifted = torch.roll(img_shifted, shifts=SHIFT, dims=2)
#     img_shifted[:,:1,:] = img_mean
#     img_shifted[:,:,:1] = img_mean
#     compare_imgs(img, img_shifted, "Shifted -")
#
#     # Set half of the image to zero
#     img_masked = img.clone()
#     img_masked[:,:img_masked.shape[1]//2,:] = img_mean
#     compare_imgs(img, img_masked, "Masked -")


# def train_cifar(latent_dim):
#     # Create a PyTorch Lightning trainer with the generation callback
#     trainer = pl.Trainer(default_root_dir='../CIFAR10' % latent_dim),
#                          checkpoint_callback=ModelCheckpoint(save_weights_only=True),
#                          gpus=1,
#                          max_epochs=500,
#                          callbacks=[GenerateCallback(get_train_images(8), every_n_epochs=10),
#                                     LearningRateMonitor("epoch")])
#     trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
#     trainer.logger._default_hp_metric = None # Optional logging argument that we don't need
#
#     # Check whether pretrained model exists. If yes, load it and skip training
#     pretrained_filename = os.path.join(CHECKPOINT_PATH, "cifar10_%i.ckpt" % latent_dim)
#     if os.path.isfile(pretrained_filename):
#         print("Found pretrained model, loading...")
#         model = Autoencoder.load_from_checkpoint(pretrained_filename)
#     else:
#         model = Autoencoder(base_channel_size=32, latent_dim=latent_dim)
#         trainer.fit(model, train_loader, val_loader)
#     # Test best model on validation and test set
#     val_result = trainer.test(model, test_dataloaders=val_loader, verbose=False)
#     test_result = trainer.test(model, test_dataloaders=test_loader, verbose=False)
#     result = {"test": test_result, "val": val_result}
#     return model, result


def visualize_reconstructions(model, input_imgs):
    # Reconstruct images
    model.eval()
    with torch.no_grad():
        reconst_imgs = model(input_imgs.to(model.device))
    reconst_imgs = reconst_imgs.cpu()

    # Plotting
    imgs = torch.stack([input_imgs, reconst_imgs], dim=1).flatten(0,1)
    grid = torchvision.utils.make_grid(imgs, nrow=4, normalize=True, range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(7,4.5))
    plt.title("Reconstructed from %i latents" % (model.hparams.latent_dim))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


def find_similar_images(query_img, query_z, key_embeds, K=8):
    # Find closest K images. We use the euclidean distance here but other like cosine distance can also be used.
    dist = torch.cdist(query_z[None,:], key_embeds[1], p=2)
    dist = dist.squeeze(dim=0)
    dist, indices = torch.sort(dist)
    # Plot K closest images
    imgs_to_display = torch.cat([query_img[None], key_embeds[0][indices[:K]]], dim=0)
    grid = torchvision.utils.make_grid(imgs_to_display, nrow=K+1, normalize=True, range=(-1,1))
    grid = grid.permute(1, 2, 0)
    plt.figure(figsize=(12,3))
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


# Plot the closest images for the first N test images as example
# for i in range(8):
#     find_similar_images(test_img_embeds[0][i], test_img_embeds[1][i], key_embeds=train_img_embeds)
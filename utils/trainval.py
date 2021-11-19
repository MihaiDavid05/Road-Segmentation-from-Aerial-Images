import torch
import os
import PIL
from tqdm import tqdm
import logging
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import optim
from torch.utils.data import DataLoader, Subset
from utils.loss import *
from utils.helpers import submission_format_metric
from torchvision import transforms
import matplotlib.image as mpimg

SEED = 45


def train(net, dataset, config, writer, device='cpu'):
    """
    Train and evaluate network.
    Args:
        net: Neural network architecture.
        dataset: Datset.
        config: Configuration dictionary.
        writer: Summary writer object for TensorBoard logging.
        device: Can be 'cpu' or 'cuda:0' - depending if you run on CPU or GPU.

    """
    # Get parameters from config
    epochs = config.epochs
    train_ratio = config.train_data_ratio
    batch_size = config.batch_size
    num_workers = config.num_workers
    learning_rate = config.learning_rate
    save_model_interval = config.save_checkpoints_interval
    weight_decay = config.weight_decay
    momentum = config.momentum
    checkpoints_path = config.checkpoints_path
    experiment_name = config.name
    patience = config.patience

    # Split dataset into train and validation subsets
    train_samples = int(len(dataset) * train_ratio)
    val_samples = len(dataset) - train_samples
    lengths = [train_samples, val_samples]
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(sum(lengths)).tolist()
    train_dataset, val_dataset = [Subset(dataset, indices[offset - length:offset])
                                  for offset, length in zip(np.cumsum(lengths), lengths)]

    # Create PyTorch DataLoaders
    loader_args = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=True, **loader_args)

    # TODO: Check this optimizer
    # Set up the optimizer, loss and learning rate scheduler
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
    logging.info(f'Training started !')
    for epoch in tqdm(range(epochs)):
        # Train step
        net.train()
        epoch_loss = 0
        for batch in train_loader:
            # Get image and groundtruth mask
            images = batch['image']
            true_masks = batch['mask']
            # Move tensors to specific device
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            # Forward pass
            optimizer.zero_grad()
            pred_masks = net(images)
            # Compute loss
            loss = criterion(pred_masks, true_masks)
            loss += dice_loss(F.softmax(pred_masks, dim=1).float(), F.one_hot(true_masks, net.n_classes).
                              permute(0, 3, 1, 2).float(), multiclass=True)
            # Write summaries to TensorBoard
            writer.add_scalar("Loss/train", loss, global_step)
            writer.add_scalar("Lr", optimizer.param_groups[0]['lr'], global_step)
            # Perform backward pass
            loss.backward()
            optimizer.step()
            # Update global step value and epoch loss
            global_step += 1
            epoch_loss += loss.item()
            # Write infos to log files
            logging.info(f'Epoch: {epoch} -> global_step: {global_step} -> train_loss: {loss.item()}')

        # Evaluate model after each epoch
        logging.info(f'Validation started !')
        val_dice_score, f1_submission = evaluate(net, val_loader, writer, epoch, device)
        # Recompute learning rate
        scheduler.step(val_dice_score)
        # Log information to logger and TensorBoard
        logging.info('Validation Dice score is: {}'.format(val_dice_score))
        logging.info('Submission score is: {}'.format(f1_submission))
        writer.add_scalar("Dice_Score/val", val_dice_score, global_step)
        writer.add_scalar("Submission_Score/val", f1_submission, global_step)
        # Save models
        if save_model_interval > 0 and (epoch + 1) % save_model_interval == 0:
            checkpoint_dir = checkpoints_path + experiment_name
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(net.state_dict(), checkpoint_dir + '/checkpoint_epoch{}.pth'.format(epoch + 1))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def evaluate(net, dataloader, writer, epoch, device='cpu'):
    """
    Run evaluation on the entire dataset.
    Args:
        net: Neural network architecture.
        dataloader: Validation dataloader.
        writer: Summary writer object for TensorBoard logging.
        epoch: Number of epoch that training has reached.
        device: Can be 'cpu' or 'cuda:0' - depending if you run on CPU or GPU.

    Returns: Validation scores (dice score and patches classification accuracy)

    """
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    f1_submission = 0

    for i, batch in tqdm(enumerate(dataloader)):
        # Get image and GT
        image = batch['image']
        mask_true = batch['mask']
        raw_mask = batch['raw_mask']
        # Log GT
        writer.add_image("GT_masks", torch.unsqueeze(mask_true[0], dim=0), i)
        # Move tensors to specific device
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
        raw_mask = raw_mask.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            # Forward pass
            pred_masks = net(image)
            if net.n_classes == 1:
                pred_masks = (F.sigmoid(pred_masks) > 0.5).float()
                # Compute dice score
                dice_score += dice_coeff(pred_masks, mask_true, reduce_batch_first=False)
            else:
                # Log prediction
                foreground_proba_pred = torch.softmax(pred_masks, dim=1)[0][1]
                writer.add_image("Pred_masks_{}".format(epoch), torch.unsqueeze(torch.softmax(pred_masks, dim=1).
                                                                                argmax(dim=1)[0],
                                                                                dim=0).float().detach().cpu(), i)
                # Compute one hot vectors
                pred_masks = F.one_hot(pred_masks.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # Compute dice score only for foreground
                dice_score += multiclass_dice_coeff(pred_masks[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)
            # TODO: Make threshold configurable
            f1_submission += submission_format_metric(foreground_proba_pred, raw_mask, fore_thresh=0.25)

    net.train()

    return dice_score / num_val_batches, f1_submission / num_val_batches


def predict_image(net,
                  full_img,
                  dataset,
                  device,
                  resize_test=True,
                  out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(dataset.preprocess(full_img, is_mask=False, is_test=True, resize_test=resize_test))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()

    if net.n_classes == 1:
        return (full_mask > out_threshold).numpy()
    else:
        return F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()


def predict(args, config, net, dataset, device):
    test_folders = os.listdir(config.test_data)
    for i, folder in enumerate(test_folders):
        filename = os.listdir(config.test_data + folder)[0]
        logging.info(f'\nPredicting image {filename}')
        img = PIL.Image.open(config.test_data + folder + '/' + filename)
        # Predict a mask
        mask = predict_image(net=net,
                             full_img=img,
                             dataset=dataset,
                             resize_test=config.resize_test,
                             out_threshold=0.5,
                             device=device)
        # TODO: Check next thing for mask and if one_hot needed

        # if not args.no_save:
        #     out_filename = out_files[i]
        #     result = mask_to_image(mask)
        #     result.save(out_filename)
        #     logging.info(f'Mask saved to {out_filename}')
        #
        # if args.viz:
        #     logging.info(f'Visualizing results for image {filename}, close to continue...')
        #     plot_img_and_mask(img, mask)

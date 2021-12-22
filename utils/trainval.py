import torch
import os
from tqdm import tqdm
import logging
import ttach as tta
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader, Subset
from utils.loss import *
from utils.helpers import mask_to_image, crop_image, overlay_masks
from torchvision import transforms
from utils.mask_to_submission import masks_to_submission


def train(net, dataset, config, writer, rng, device='cpu'):
    """
    Train and evaluate network.
    Args:
        net: Neural network architecture.
        dataset: Dataset instance.
        config: Configuration dictionary
        writer: Summary writer object for TensorBoard logging
        rng: Random number generator
        device: Can be 'cpu' or 'cuda:0' - depending if you run on CPU or GPU.

    """
    # Get parameters from config
    epochs = config.epochs
    train_ratio = config.train_data_ratio
    batch_size = config.batch_size
    num_workers = config.num_workers
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    momentum = config.momentum
    checkpoints_path = config.checkpoints_path
    experiment_name = config.name
    patience = config.patience

    # Split dataset into train and validation subsets
    train_samples = int(len(dataset) * train_ratio)
    val_samples = len(dataset) - train_samples
    lengths = [train_samples, val_samples]
    indices = rng.permutation(sum(lengths)).tolist()
    train_dataset, val_dataset = [Subset(dataset, indices[offset - length:offset])
                                  for offset, length in zip(np.cumsum(lengths), lengths)]

    # Create PyTorch DataLoaders
    loader_args = dict(num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, drop_last=True, **loader_args)

    # Set up the optimizer and learning rate scheduler
    if config.optim_type == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=weight_decay, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=patience)

    # Set loss
    if config.loss_type == 'focal':
        criterion = FocalLoss(gamma=config.focal_gamma, alpha=config.focal_alpha)
    else:
        criterion = nn.CrossEntropyLoss()

    # Initialize variables
    global_step = 0
    max_val_score = 0
    logging.info(f'Training started !')
    # Train
    for epoch in tqdm(range(epochs)):
        # Train step
        net.train()
        epoch_loss = 0
        for batch in train_loader:
            # Get image and groundtruth mask
            images = batch['image']
            binary_mask = batch['mask']
            # Move tensors to specific device
            images = images.to(device=device, dtype=torch.float32)
            binary_mask = binary_mask.to(device=device, dtype=torch.long)
            # Forward pass
            optimizer.zero_grad()
            pred_masks = net(images)
            # Compute loss
            loss = criterion(pred_masks, binary_mask)
            loss += dice_loss(F.softmax(pred_masks, dim=1).float(), F.one_hot(binary_mask, net.n_classes).
                              permute(0, 3, 1, 2).float(), multiclass=True)
            # Write summaries to TensorBoard
            writer.add_scalar("Lr", optimizer.param_groups[0]['lr'], global_step)
            # Perform backward pass
            loss.backward()
            optimizer.step()
            # Update global step value and epoch loss
            global_step += 1
            epoch_loss += loss.item()
        epoch_loss = epoch_loss / len(train_loader)
        # Write info to log files
        logging.info(f'Epoch: {epoch} -> train_loss: {epoch_loss}')

        # Evaluate model after each epoch
        logging.info(f'Validation started !')
        val_dice_score, val_loss = evaluate(net, val_loader, writer, epoch, criterion, device)
        # Plot val loss for training and validation
        writer.add_scalars('Loss', {'train': epoch_loss, 'val': val_loss}, global_step)
        # Update learning rate
        scheduler.step(val_dice_score)
        # Log information to logger and TensorBoard
        logging.info('Validation Dice score is: {}'.format(val_dice_score))
        writer.add_scalar("Dice_Score/val", val_dice_score, global_step)

        # Save checkpoint after each epoch
        checkpoint_dir = checkpoints_path + experiment_name
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(net.state_dict(), checkpoint_dir + '/checkpoint_' + str(epoch) + '.pth')

        # Save best model by highest validation score)
        if val_dice_score > max_val_score:
            max_val_score = val_dice_score
            print("Current maximum validation score is: {}".format(max_val_score))
            torch.save(net.state_dict(), checkpoint_dir + '/checkpoint_BEST.pth')
            logging.info(f'Checkpoint {epoch} saved!')


def evaluate(net, dataloader, writer, epoch, criterion, device='cpu'):
    """
    Run evaluation on the entire dataset.
    Args:
        net: Neural network architecture.
        dataloader: Validation dataloader.
        writer: Summary writer object for TensorBoard logging.
        epoch: Number of epoch that training has reached.
        criterion: Loss criterion.
        device: Can be 'cpu' or 'cuda:0' - depending if you run on CPU or GPU.

    Returns: Validation score and loss

    """
    net.eval()
    # Initialize varibales
    num_val_batches = len(dataloader)
    dice_score = 0
    val_loss = 0
    for i, batch in tqdm(enumerate(dataloader)):
        # Get image and gt masks
        image = batch['image']
        binary_mask = batch['mask']
        # Log ground truth
        writer.add_image("GT_masks", torch.unsqueeze(binary_mask[0], dim=0), i)
        # Move tensors to specific device
        image = image.to(device=device, dtype=torch.float32)
        binary_mask = binary_mask.to(device=device, dtype=torch.long)
        binary_mask_one_hot = F.one_hot(binary_mask, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # Forward pass
            pred_masks = net(image)
            if net.n_classes == 1:
                # Compute dice score
                pred_masks = (F.sigmoid(pred_masks) > 0.5).float()
                dice_score += dice_coeff(pred_masks, binary_mask_one_hot, reduce_batch_first=False)
            else:
                # Log prediction to tensorboard
                writer.add_image("Pred_masks_{}".format(epoch), torch.unsqueeze(torch.softmax(pred_masks, dim=1).
                                                                                argmax(dim=1)[0],
                                                                                dim=0).float().detach().cpu(), i)
                # Compute validation loss
                loss = criterion(pred_masks, binary_mask)
                loss += dice_loss(F.softmax(pred_masks, dim=1).float(), F.one_hot(binary_mask, net.n_classes).
                                  permute(0, 3, 1, 2).float(), multiclass=True)
                val_loss += loss.item()
                # Compute one hot vectors
                pred_masks = F.one_hot(pred_masks.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # Compute dice score only for foreground pixels
                dice_score += multiclass_dice_coeff(pred_masks[:, 1:, ...], binary_mask_one_hot[:, 1:, ...],
                                                    reduce_batch_first=False)
    net.train()
    # Update and log validation loss
    val_loss = val_loss / len(dataloader)
    logging.info(f'Epoch: {epoch} -> val_loss: {val_loss}')

    return dice_score / num_val_batches, val_loss


def predict_image(net,
                  initial_img,
                  dataset,
                  device,
                  resize_test=True,
                  out_threshold=0.5,
                  test_time_aug=False):
    """
    Make prediction on a patch.
    Args:
        net: Network
        initial_img: Entire image or a image patch
        dataset: Test dataset instance
        device: "cpu" or "cuda:0"
        resize_test: Whether to resize the test image after prediction
        out_threshold: Threshold if a single class is used
        test_time_aug: Whether to use test time augmentation or not

    Returns: Probabilities and one-hot mask for initial_img

    """
    net.eval()
    # Pre process test image
    img = torch.from_numpy(dataset.preprocess(initial_img, is_mask=False, is_test=True, resize_test=resize_test))
    img = img.unsqueeze(0)
    # Send to device
    img = img.to(device=device, dtype=torch.float32)

    # Define test time augmentations
    transformations_test_time_aug = tta.Compose(
        [tta.HorizontalFlip(), tta.VerticalFlip(), tta.Rotate90(angles=[0, 90, 180, 270])]
    )
    with torch.no_grad():
        if test_time_aug:
            labels = []
            # Iterate through types of test augmentations
            for transformer in transformations_test_time_aug:
                # Augment image
                augmented_image = transformer.augment_image(img)
                # Predict
                model_output = net(augmented_image)
                # Reverse augmentation for label
                deaug_label = transformer.deaugment_mask(model_output)
                # Store results
                labels.append(deaug_label)
            # Reduce results
            bg_mask = torch.cat([t[:, 0, ...] for t in labels], dim=0)
            bg_mask = torch.sum(bg_mask, dim=0) / bg_mask.size(0)
            fg_mask = torch.cat([t[:, 1, ...] for t in labels], dim=0)
            fg_mask = torch.sum(fg_mask, dim=0) / fg_mask.size(0)
            output = torch.cat([bg_mask.unsqueeze(dim=0), fg_mask.unsqueeze(dim=0)], dim=0).unsqueeze(dim=0)
        else:
            # Predict without test time augmentations
            output = net(img)
        # Get probabilities masks
        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        if resize_test:
            # Resize test image from 400x400 to 608x608
            tf = transforms.Compose(
                [transforms.ToPILImage(), transforms.Resize((initial_img.size[1], initial_img.size[0])),
                 transforms.ToTensor()]
            )
            proba_mask = tf(probs.cpu()).squeeze()
        else:
            proba_mask = probs.cpu().squeeze()

        # Get one-hot masks
        if net.n_classes == 1:
            one_hot_mask = (proba_mask > out_threshold).numpy()
        else:
            one_hot_mask = F.one_hot(proba_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()

    return proba_mask.numpy(), one_hot_mask


def predict(args, config, net, dataset, device):
    """
    Get predictions and create submission file
    Args:
        args: Command line arguments dictionary
        config: Config dictionary
        net: Network
        dataset: Test dataset instance
        device: Either "cpu" or "cuda:0"

    """
    # Get test data
    test_folders = os.listdir(config.test_data)
    test_folders = sorted(test_folders, key=lambda x: int(x.split('_')[-1]))
    # Set predictions visualization folder
    viz_folder = config.viz_path + config.name
    if not os.path.exists(viz_folder):
        os.mkdir(viz_folder)
    # TODO: Here remove after aug
    # Set submission file
    submission_path = config.output_path + "submission_" + config.name + '_' + config.patch_combine + '_patch_ttime_aug_checkpoint_17' + '.csv'
    preds = []
    # Iterate through test images and make predictions
    for i, folder in tqdm(enumerate(test_folders)):
        filename = os.listdir(config.test_data + folder)[0]
        logging.info(f'\nPredicting image {filename}')
        # Read test image
        img = Image.open(config.test_data + folder + '/' + filename)
        if config.predict_patches:
            # Get 4 400x400 patches
            patches = crop_image(img, 400, 400)
        else:
            # Keep entire image
            patches = [img]

        proba_masks = []
        one_hot_masks = []
        # For each sub image (patch), get a prediction
        for patch in patches:
            proba_mask, one_hot_mask = predict_image(net=net,
                                                     initial_img=patch,
                                                     dataset=dataset,
                                                     resize_test=config.resize_test,
                                                     out_threshold=0.5,
                                                     test_time_aug=config.test_time_aug,
                                                     device=device)
            # Gather probabilities and one-hot masks
            proba_masks.append(proba_mask)
            one_hot_masks.append(one_hot_mask)

        if len(proba_masks) > 1:
            # Multiple patches predictions
            proba_mask = overlay_masks(proba_masks, img, mode=config.patch_combine)
            one_hot_mask = F.one_hot(torch.tensor(proba_mask,
                                                  device='cpu').argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
        else:
            # Single image prediction
            proba_mask = proba_masks[0]
            one_hot_mask = one_hot_masks[0]

        # Append foreground mask to predictions
        foreground_mask = proba_mask[1]
        preds.append((filename, foreground_mask))

        # Visualize prediction
        if args.save:
            out_filename = test_folders[i] + '_from_patches.png'
            output_path = viz_folder + '/' + out_filename
            # Get one hot mask and transform it to image
            out_mask = one_hot_mask[1]
            result = mask_to_image(out_mask, from_patches=len(patches) > 1)
            # Save image
            result.save(output_path)
            logging.info(f'Mask saved to {out_filename}')

    # Convert to submission format and save to csv
    masks_to_submission(submission_path, config.foreground_thresh, preds)

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset


def train(net, dataset, config, device='cpu'):
    # Get parameters from config
    epochs = config.epochs
    train_ratio = config.train_data_ratio
    batch_size = config.batch_size
    num_workers = config.num_workers

    # Split dataset into train and validation subsets
    train_samples = int(len(dataset) * train_ratio)
    val_samples = len(dataset) - train_samples
    lengths = [train_samples, val_samples]
    # Set random seed and get indices as permutations
    rng = np.random.RandomState(45)
    indices = rng.permutation(sum(lengths)).tolist()
    train_dataset, val_dataset = [Subset(dataset, indices[offset - length:offset])
                                  for offset, length in zip(np.cumsum(lengths), lengths)]

    # Create PyTorch DataLoaders
    loader_args = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=True, **loader_args)

    # TODO: Check this
    # # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # criterion = nn.CrossEntropyLoss()
    # global_step = 0

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        for batch in train_loader:
            images = batch['image']
            images_numpy = images.numpy()
            true_masks = batch['mask']
            # Move tensors to GPU
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            # Forward pass
            masks_pred = net(images)
            # TODO: Check this
            loss = 0
            print("OK")

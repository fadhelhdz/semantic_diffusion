import torch
import torch.nn as nn
import torch.optim as optim
from codes.calculate.noise import add_awgn_noise


def train_semantic_communication_system(channel, dataloader, num_epochs, device, train_snr, lr=0.001):
    channel = channel.to(device)
    encoder = channel.encoder
    decoder = channel.decoder

    # Move the channel to the selected device
    criterion = nn.MSELoss()
    optimizer = optim.Adam(channel.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        for batch_images in dataloader:
            batch_images = batch_images[0].to(device)

            optimizer.zero_grad()
            encoded_images = encoder(batch_images)
            
            # Add noise into training encoded images
            noisy_images = add_awgn_noise(encoded_images, train_snr)
            noisy_images = noisy_images.to(device)

            # Data restoration
            restored_images = decoder(noisy_images)

            # Calculate loss
            loss = criterion(restored_images, batch_images)

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

        # Print progress
        if (epoch + 1) % 50 == 0:
            print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    return encoder, decoder


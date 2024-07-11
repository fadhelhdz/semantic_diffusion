import torch.nn as nn

# Define semantic encoder and decoder models
class SemanticCommunicationChannel(nn.Module):
    def __init__(self):
        super(SemanticCommunicationChannel, self).__init__()
        self.encoder = SemanticEncoder()
        self.decoder = SemanticDecoder()

    def forward(self, x):
        # Encode the input image into a latent representation
        encoded = self.encoder(x)

        # Send the encoded representation through the channel
        transmitted = encoded

        # Decode the transmitted representation back into the semantic space
        decoded = self.decoder(transmitted)

        return decoded


class SemanticEncoder(nn.Module):
    def __init__(self):
        super(SemanticEncoder, self).__init__()

        # Define encoder architecture
        self.batchnorm1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu1 = nn.PReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.relu2 = nn.PReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu3 = nn.PReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu4 = nn.PReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=1)
        
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.relu5 = nn.PReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.batchnorm5 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.batchnorm1(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = self.batchnorm5(x)

        return x


class SemanticDecoder(nn.Module):
    def __init__(self):
        super(SemanticDecoder, self).__init__()

        # Define decoder architecture
        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.relu1 = nn.PReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.relu2 = nn.PReLU()

        self.deconv3 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=1, padding=1)
        self.relu3 = nn.PReLU()

        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.relu4 = nn.PReLU()

        self.deconv5 = nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1)
        self.relu5 = nn.PReLU()

        self.batchnorm5 = nn.BatchNorm2d(3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.relu2(x)

        x = self.deconv3(x)
        x = self.relu3(x)

        x = self.deconv4(x)
        x = self.relu4(x)

        x = self.deconv5(x)
        x = self.relu5(x)

        x = self.batchnorm5(x)
        x = self.sigmoid(x)

        return x

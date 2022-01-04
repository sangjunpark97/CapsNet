import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
import os
import torchvision.utils as vutils
import tensorflow as tf

class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CapsuleConvLayer, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=9, # fixme constant
                               stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv0(x))

class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.Conv2d(in_channels=in_channels,
                               out_channels=32,  # fixme constant
                               kernel_size=9,  # fixme constant
                               stride=2, # fixme constant
                               bias=True)

    def forward(self, x):
        return self.conv0(x)

class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing

        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))
        else:
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # Instead, it is composed of several convolutional units, each of which sees the full input.
            # It is implemented as a normal convolutional layer with a special nonlinearity (squash()).
            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, height, width).
        u = [self.units[i](x) for i in range(self.num_units)]

        # Stack all unit outputs (batch, unit, channels, height, width).
        u = torch.stack(u, dim=1)

        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.num_units, -1) # num_units = 출력 개수로 본문에서는 10개를 사용한다.

        # Return squashed outputs.
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1))

        # Iterative routing.
        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij)
            
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)
            print(c_ij[0][0])
            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            
            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)         
            # [ 10  16  ]
            # (batch_size, features, num_units, unit_size, 1)

            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)
            
            # [  ]
            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)
            
            #  [  1152   10   ]
            # Update b_ij (routing)
            b_ij = b_ij + u_vj1


        return v_j.squeeze(1)

class CapsuleNetwork(nn.Module):
    def __init__(self,
                 image_width,
                 image_height,
                 image_channels,
                 conv_inputs,
                 conv_outputs,
                 num_primary_units,
                 primary_unit_size,
                 num_output_units,
                 output_unit_size):
        super(CapsuleNetwork, self).__init__()

        self.reconstructed_image_count = 0

        self.image_channels = image_channels
        self.image_width = image_width
        self.image_height = image_height

        self.conv1 = CapsuleConvLayer(in_channels=conv_inputs,
                                      out_channels=conv_outputs)

        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=conv_outputs,
                                    num_units=num_primary_units,
                                    unit_size=primary_unit_size,
                                    use_routing=False)

        self.digits = CapsuleLayer(in_units=num_primary_units,
                                   in_channels=primary_unit_size,
                                   num_units=num_output_units,
                                   unit_size=output_unit_size,
                                   use_routing=True)

        reconstruction_size = image_width * image_height * image_channels
        self.reconstruct0 = nn.Linear(num_output_units*output_unit_size, int((reconstruction_size * 2) / 3))
        self.reconstruct1 = nn.Linear(int((reconstruction_size * 2) / 3), int((reconstruction_size * 3) / 2))
        self.reconstruct2 = nn.Linear(int((reconstruction_size * 3) / 2), reconstruction_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.digits(self.primary(self.conv1(x)))

    def loss(self, images, input, target, size_average=True):
        return self.margin_loss(input, target, size_average) + self.reconstruction_loss(images, input, size_average)

    def margin_loss(self, input, target, size_average=True):
        batch_size = input.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((input**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1))
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    def reconstruction_loss(self, images, input, size_average=True):
        # Get the lengths of capsule outputs.
        v_mag = torch.sqrt((input**2).sum(dim=2))

        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data

        # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
        batch_size = input.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            # Get one sample from the batch.
            input_batch = input[batch_idx]

            # Copy only the maximum capsule index from this batch sample.
            # This masks out (leaves as zero) the other capsules in this sample.
            batch_masked = Variable(torch.zeros(input_batch.size()))
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked

        # Stack masked capsules over the batch dimension.
        masked = torch.stack(all_masked, dim=0)

        # Reconstruct input image.
        masked = masked.view(input.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        output = output.view(-1, self.image_channels, self.image_height, self.image_width)

        # Save reconstructed images occasionally.
        if self.reconstructed_image_count % 10 == 0:
            if output.size(1) == 2:
                # handle two-channel images
                zeros = torch.zeros(output.size(0), 1, output.size(2), output.size(3))
                output_image = torch.cat([zeros, output.data.cpu()], dim=1)
            else:
                # assume RGB or grayscale
                output_image = output.data.cpu()
            vutils.save_image(output_image, "reconstruction.png")
        self.reconstructed_image_count += 1

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (output - images).view(output.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1) * 0.0005

        # Average over batch
        if size_average:
            error = error.mean()

        return error
        
learning_rate = 0.01
batch_size = 256
test_batch_size = 128

# Stop training if loss goes below this threshold.
early_stop_loss = 0.0001

#
# Load MNIST dataset.
#

# Normalization for MNIST dataset.
dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])

train_dataset = datasets.MNIST('C:/Users/Jet Zhang/Desktop/pytorch/GAN/mnist', train=True, download=True, transform=dataset_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST('C:/Users/Jet Zhang/Desktop/pytorch/GAN/mnist', train=False, download=True, transform=dataset_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

#
# Create capsule network.
#

conv_inputs = 1
conv_outputs = 256
num_primary_units = 8
primary_unit_size = 32 * 6 * 6  # fixme get from conv2d
output_unit_size = 16

network = CapsuleNetwork(image_width=28,
                         image_height=28,
                         image_channels=1,
                         conv_inputs=conv_inputs,
                         conv_outputs=conv_outputs,
                         num_primary_units=num_primary_units,
                         primary_unit_size=primary_unit_size,
                         num_output_units=10, # one for each MNIST digit
                         output_unit_size=output_unit_size)
                         
                         
# Converts batches of class indices to classes of one-hot vectors.
def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot

# This is the test function from the basic Pytorch MNIST example, but adapted to use the capsule network.
# https://github.com/pytorch/examples/blob/master/mnist/main.py
def test():
    network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target_indices = target
        target_one_hot = to_one_hot(target_indices, length=network.digits.num_units)

        data, target = Variable(data, volatile=True), Variable(target_one_hot)

        output = network(data)

        test_loss += network.loss(data, output, target).item() # sum up batch loss

        v_mag = torch.sqrt((output**2).sum(dim=2, keepdim=True))

        pred = v_mag.data.max(1, keepdim=True)[1].cpu()

        correct += pred.eq(target_indices.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# This is the train function from the basic Pytorch MNIST example, but adapted to use the capsule network.
# https://github.com/pytorch/examples/blob/master/mnist/main.py
def train(epoch):
    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    last_loss = None
    log_interval = 1
    network.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        target_one_hot = to_one_hot(target, length=network.digits.num_units)

        data, target = Variable(data), Variable(target_one_hot)

        optimizer.zero_grad()

        output = network(data)

        loss = network.loss(data, output, target)
        loss.backward()
        last_loss = loss.item()

        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),100. * batch_idx / len(train_loader),
                loss.item()))
        
        if last_loss < early_stop_loss:
            break
    
    return last_loss

num_epochs = 1


for epoch in range(1, num_epochs + 1):
    last_loss = train(epoch)
    
    if last_loss < early_stop_loss:
        break

with torch.no_grad():
            test()

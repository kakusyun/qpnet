import torch.nn as nn

class VGG16_Backbone(nn.Module):

    def __init__(self, input_channel, output_channel=512):
        super(VGG16_Backbone, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[0], self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[1], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            # Block 4
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),
            nn.ReLU(True),
            # Block 5
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1),
            nn.ReLU(True),

            nn.BatchNorm2d(self.output_channel[3]))

    def forward(self, input):
        return self.ConvNet(input)

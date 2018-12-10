import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2)))
        self.list.append(self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2)))
        self.list.append(self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2)))
        self.list.append(self._make_conv_layer(256, 512, (2, 2, 2), (2, 2, 2)))
        for i in range(1):
            self.list.append(self._make_conv_layer(512, 512, (1, 1, 1), (1, 1, 1)))
        #self.list.append(self._make_conv_layer(in_c=1024, out_c=512, pool_size=(1, 1, 2), stride=(1, 1, 1)))
        """after poooling there is a change of the dimention of the output data, according to MaxPool3d
            Out = (in - kernel_size + 2*padding)/ stride + 1
            and it is calculated for each dimenstion, without batch and depth (RGB, probably 3)
        """

    def _make_conv_layer(self, in_c, out_c, pool_size, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ELU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        for i, module in enumerate(self.list):
            x = module(x)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm_layer1 = nn.LSTM(input_size=50, hidden_size=50,
                                   num_layers=2, batch_first=True)

    def forward(self, x):
        x = self.lstm_layer1(x)
        return x


class FC(nn.Module):
    def __init__(self, num_classes):
        super(FC, self).__init__()
        self.list = nn.ModuleList()
        self.list.append(nn.Linear(in_features=25600, out_features=512, bias=True))
        self.fc_layer2_act = nn.ELU()
        self.fc_layer3 = nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x):
        for i, module in enumerate(self.list):
            x = module(x)
        x = self.fc_layer2_act(x)
        x = self.fc_layer3(x)
        return x


class MyNetwork(nn.Module):

    def __init__(self, num_classes, show = False):
        self.show = show
        super(MyNetwork, self).__init__()
        self.CNN = CNN()
        self.LSTM = LSTM()
        self.FC = FC(num_classes)

    def forward(self, x):
        if self.show:
            print("CNN in: \t", x.size())
        x = self.CNN(x)
        if self.show:
            print("CNN out: \t", x.size())
        x = x.view(x.size(0), x.size(1), -1)
        if self.show:
            print("LSTM in: \t", x.size())
        x, (hidden, cell) = self.LSTM(x)
        if self.show:
            print("LSTM out: \t", x.size())
        x = x.contiguous()
        x = x.view(x.size(0), -1)
        if self.show:
            print("FC in:  \t", x.size())
        x = self.FC(x)
        if self.show:
            print("FC out: \t", x.size(), '\n')
        return x


if __name__ == "__main__":
    input_tensor = torch.autograd.Variable(torch.rand(5, 3, 18, 84, 84))
    model = MyNetwork(10, show=True)  # ConvColumn(27).cuda()
    output = model(input_tensor)  # model(input_tensor.cuda())
    print(model)

import sys

import torch
from torch import nn
from torch.nn import init


########################################
#        Zhenchun Lei
#  zhenchun.lei@hotmail.com
########################################


class MultiLayerClassifier(nn.Module):
    def __init__(self, layer_channels=[512], dropout=0.0):
        super(MultiLayerClassifier, self).__init__()
        # self.input_size = input_size
        self.layer_channels = layer_channels

        classifier = []

        if len(layer_channels) == 2:
            classifier.append(nn.Linear(layer_channels[0], layer_channels[1]))
            # classifier.append(nn.ReLU())
            self.classifier = nn.Sequential(*classifier)
        else:
            for idx in range(0, len(layer_channels) - 2):
                classifier.append(nn.Linear(layer_channels[idx], layer_channels[idx + 1]))
                classifier.append(nn.ReLU())
                if dropout > 0:
                    classifier.append(nn.Dropout(p=dropout))
            classifier.append(nn.Linear(layer_channels[-2], layer_channels[-1]))
            self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        return self.classifier(x)


class TwoPathModel(nn.Module):
    def __init__(self, path1, path2, classifier, path_input_size=0):
        super(TwoPathModel, self).__init__()
        self.path1 = path1
        self.path2 = path2
        self.classifier = classifier
        # self.path_input_size = path_input_size

    def forward(self, x):
        # x1 = x[:, :self.path_input_size, :]
        # x2 = x[:, self.path_input_size:, :]

        x1 = self.path1(x)
        x2 = self.path2(x)

        output = torch.cat((x1, x2), dim=1)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return output


class MPathModel(nn.Module):
    def __init__(self, paths, classifier, path_inputs=0, path_outputs=0):
        super(MPathModel, self).__init__()
        self.paths = nn.ModuleList(paths)
        self.classifier = classifier
        # self.path_inputs = path_inputs
        # self.path_outputs = path_outputs
        self.path_num = len(self.paths)

    def forward(self, x):
        # batch_size, channels, time_len = x.shape

        # output = torch.cuda.FloatTensor(batch_size, self.path_outputs * self.path_num)
        # for idx in range(self.path_num):
        #     # output[:, self.path_outputs * idx:self.path_outputs * (idx + 1)] = self.paths[idx](
        #     #     x[:, self.path_inputs * idx:self.path_inputs * (idx + 1), :])
        #     output[:, self.path_outputs * idx:self.path_outputs * (idx + 1)] = self.paths[idx](x)

        output = []
        for path in self.paths:
            # output[:, self.path_outputs * idx:self.path_outputs * (idx + 1)] = self.paths[idx](
            #     x[:, self.path_inputs * idx:self.path_inputs * (idx + 1), :])
            output.append(path(x))

        # y = []
        # y = torch.zeros(())
        # for idx in range(self.path_num):
        #     y.append(self.paths[idx](x[:, self.path_inputs * idx:self.path_inputs * (idx + 1), :]))
        # output = torch.cat(y, dim=1)

        # output = output.view(output.size(0), -1)
        output = torch.cat(output, dim=1)
        output = output.view(output.size(0), -1)
        output = self.classifier(output)

        return output


class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, drop_out=0.5):
        super(DNN, self).__init__()
        self.num_classes = num_classes
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),

            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(drop_out),

            nn.Linear(hidden_size, num_classes),
            # nn.Softmax(dim=1)
        )

        self.init_weight(self.main)

    def forward(self, x):
        output = self.main(x)
        return output

    def init_weight(self, m):
        for each_module in m:
            if "Linear" in each_module.__class__.__name__:
                init.xavier_normal_(each_module.weight)
                init.constant_(each_module.bias, 0.)


class TDNN(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride=1, padding=0, dilation=1):
        super(TDNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=output_size, kernel_size=(input_size, kernel_size),
                              stride=stride, padding=padding, dilation=dilation)

        init.xavier_normal_(self.conv.weight)
        init.constant_(self.conv.bias, 0.)

    def forward(self, x):
        output = self.conv(x)
        output = output.view(output.size(0), 1, output.size(1), output.size(3))
        return output


class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D)

    MaxFeatureMap2D(max_dim=1)

    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)


    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)

    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)

    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1):
        super().__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
            sys.exit(1)
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
            sys.exit(1)
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m


class AdaptiveStdMeanPool1d(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, inputs):
        return torch.cat(torch.std_mean(inputs, dim=2), dim=1)

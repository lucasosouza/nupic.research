from torch import nn
import torch

class CustomMLP(nn.Module):

    def __init__(self, config={}):

        super(CustomMLP, self).__init__()

        defaults = dict(
            input_size=784, 
            num_classes=10, 
            hidden_sizes=[300, 300, 300], # smaller to run fast
            batch_norm=False, 
            dropout=0.5, 
            bias=False, 
            init_weights=True, 
        )
        defaults.update(config)
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        layers = []
        self.conv1 = self.linear_block(
            self.input_size, self.hidden_sizes[0], bn=self.batch_norm, dropout=self.dropout, bias=self.bias)
        self.conv2 = self.linear_block(
            self.hidden_sizes[0], self.hidden_sizes[1], bn=self.batch_norm, dropout=self.dropout, bias=self.bias)
        self.conv3 = self.linear_block(
            self.hidden_sizes[1], self.hidden_sizes[2], bn=self.batch_norm, dropout=self.dropout, bias=self.bias)

        self.conv_block = nn.Sequential(*layers)
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.output_layer = nn.Linear(self.hidden_sizes[2], self.num_classes, bias=self.bias)

        if self.init_weights:
            self._initialize_weights(self.bias)

    @staticmethod
    def linear_block(a, b, bn=False, dropout=False, bias=True):
        block = [nn.Linear(a, b, bias=bias), nn.ReLU()]
        if bn:
            block.append(nn.BatchNorm1d(b))
        return nn.Sequential(*block)

    def forward(self, x, set_dropout=True):
        x = x.view(-1, self.input_size)
        x = self.conv1(x)
        if set_dropout: x = self.dropout_layer(x)
        x = self.conv2(x)
        if set_dropout: x = self.dropout_layer(x)
        x = self.conv3(x)
        if set_dropout: x = self.dropout_layer(x)

        x = self.output_layer(x)
        return x

    def _initialize_weights(self, bias):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # regular xavier initialization
                if bias:
                    nn.init.constant_(m.bias, 0)


class RegularMLP(nn.Module):

    def __init__(self, config={}):

        super(RegularMLP, self).__init__()

        defaults = dict(
            input_size=784, 
            num_classes=10, 
            hidden_sizes=[300, 300, 300], # smaller to run fast
            batch_norm=False, 
            dropout=0.5, 
            bias=False, 
            init_weights=True, 
        )
        defaults.update(config)
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        layers = []
        self.conv1 = self.linear_block(
            self.input_size, self.hidden_sizes[0], bn=self.batch_norm, dropout=self.dropout, bias=self.bias)
        self.conv2 = self.linear_block(
            self.hidden_sizes[0], self.hidden_sizes[1], bn=self.batch_norm, dropout=self.dropout, bias=self.bias)
        self.conv3 = self.linear_block(
            self.hidden_sizes[1], self.hidden_sizes[2], bn=self.batch_norm, dropout=self.dropout, bias=self.bias)

        self.output_layer = nn.Linear(self.hidden_sizes[2], self.num_classes, bias=self.bias)

        if self.init_weights:
            self._initialize_weights(self.bias)

    @staticmethod
    def linear_block(a, b, bn=False, dropout=False, bias=True):
        block = [nn.Linear(a, b, bias=bias), nn.ReLU()]
        if bn:
            block.append(nn.BatchNorm1d(b))
        if dropout:
            block.append(nn.Dropout(p=dropout))
        return nn.Sequential(*block)

    def forward(self, x, set_dropout=True):
        x = x.view(-1, self.input_size)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self, bias):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) # regular xavier initialization
                if bias:
                    nn.init.constant_(m.bias, 0)



class CNN(nn.Module):
    """ 
    Simple implementenation of CNN
    """

    def __init__(self, config={}):

        super(MLP, self).__init__()

        defaults = dict(
            input_size=784,
            num_classes=10,
            hidden_sizes=[4000, 1000, 4000],
            batch_norm=False,
            dropout=0.3,
            bias=False,
            init_weights=True,
        )
        defaults.update(config)
        self.__dict__.update(defaults)
        self.device = torch.device(self.device)

        layers = []
        layers.extend(
            self.linear_block(
                self.input_size,
                self.hidden_sizes[0],
                bn=self.batch_norm,
                dropout=self.dropout,
                bias=self.bias,
            )
        )
        layers.extend(
            self.linear_block(
                self.hidden_sizes[0],
                self.hidden_sizes[1],
                bn=self.batch_norm,
                dropout=self.dropout,
                bias=self.bias,
            )
        )
        layers.extend(
            self.linear_block(
                self.hidden_sizes[1],
                self.hidden_sizes[2],
                bn=self.batch_norm,
                dropout=self.dropout,
                bias=self.bias,
            )
        )

        # output layer
        layers.append(nn.Linear(self.hidden_sizes[2], self.num_classes, bias=self.bias))
        self.classifier = nn.Sequential(*layers)

        if self.init_weights:
            self._initialize_weights(self.bias)

    @staticmethod
    def linear_block(a, b, bn=False, dropout=False, bias=True):
        block = [nn.Linear(a, b, bias=bias), nn.ReLU()]
        if bn:
            block.append(nn.BatchNorm1d(b))
        if dropout:
            block.append(nn.Dropout(p=dropout))

        return block

    def forward(self, x):
        return self.classifier(x.view(-1, self.input_size))

    def _initialize_weights(self, bias):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if bias:
                    nn.init.constant_(m.bias, 0)


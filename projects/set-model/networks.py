from torch import nn

class MLP(nn.Module):

    def __init__(self, config={}):

        super(MLP, self).__init__()

        defaults = dict(
            input_size=784, 
            num_classes=10, 
            hidden_sizes=[1000, 1000, 1000], 
            batch_norm=False, 
            dropout=False, 
            bias=True, 
            init_weights=True, 
        )
        defaults.update(config)
        self.__dict__.update(defaults)

        layers = []
        layers.extend(self.linear_block(
            self.input_size, self.hidden_sizes[0], bn=self.batch_norm, bias=self.bias))
        layers.extend(self.linear_block(
            self.hidden_sizes[0], self.hidden_sizes[1], bn=self.batch_norm, bias=self.bias))
        layers.extend(self.linear_block(
            self.hidden_sizes[1], self.hidden_sizes[2], bn=self.batch_norm, bias=self.bias))

        if self.dropout:
            layers.append(nn.Dropout(p=0.3))
        
        # output layer
        layers.append(nn.Linear(self.hidden_sizes[2], self.num_classes, bias=self.bias))
        self.classifier = nn.Sequential(*layers)

        if self.init_weights:
            self._initialize_weights(self.bias)

    @staticmethod
    def linear_block(a, b, bn=False, bias=True):
        block = [nn.Linear(a, b, bias=bias), nn.ReLU()]
        if bn:
            block.append(nn.BatchNorm1d(b))

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
                nn.init.normal_(m.weight, 0, 0.01)
                if bias:
                    nn.init.constant_(m.bias, 0)


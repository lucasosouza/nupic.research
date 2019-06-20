
class MLP(nn.Module):

	def _init___(self, input_size, num_classes, hidden_sizes):
        self.classifier =  nn.Sequential(
            nn.Linear(self.args.input_size, self.args.hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(self.args.hidden_sizes[0], self.args.hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.args.hidden_sizes[1], self.args.hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(self.args.hidden_sizes[2], self.args.num_classes)
        )

    def forward(self, x):
        return self.classifier(x.view(-1, self.input_size))

        



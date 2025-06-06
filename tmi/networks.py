import torch.nn as nn
import torch.nn.functional as F

class TeacherNetwork(nn.Module):
    def __init__(self):
        super(TeacherNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.dropout_input = 0.2
        self.dropout_hidden = 0.5
        self.is_training = True

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden, training=self.is_training)
        x = F.dropout(F.relu(self.fc2(x)), p=self.dropout_hidden, training=self.is_training)
        x = self.fc3(x)
        return x

class StudentNetwork(nn.Module):
    def __init__(self):
        super(StudentNetwork, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 250)
        self.fc2 = nn.Linear(250, 10)
        self.dropout_input = 0.1
        self.dropout_hidden = 0.2
        self.is_training = True

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.dropout(x, p=self.dropout_input, training=self.is_training)
        x = F.dropout(F.relu(self.fc1(x)), p=self.dropout_hidden, training=self.is_training)
        x = self.fc2(x)
        return x

from preprocessing import *
import torch.nn as nn
import torch 
import torch.optim as optim
import torch.utils.data as data_utils
from torchsummary import summary
from torchmetrics.functional import f1_score

# Recurrent neural network (many-to-one)
class RNN_Predictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN_Predictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)

        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0

    # Set model to eval
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device).squeeze(1)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    # Toggle model back to train
    #model.train()
    return num_correct / num_samples

#Load Data from files
[features, target] = file_parser(0)

X = features
Y= target


#Training data
X_train = X[0:50]
Y_train = Y[0:50]
x_train = torch.FloatTensor(np.array(X_train))
y_train = torch.LongTensor(Y_train)

#Test data
X_test  = X[50:]
Y_test = Y[50:]
x_test = torch.FloatTensor(np.array(X_test))
y_test = torch.LongTensor(Y_test)
print(x_train.shape)
print(y_test.shape)

#Load Data
train = data_utils.TensorDataset(x_train, y_train)
test = data_utils.TensorDataset(x_test, y_test)
train_loader = data_utils.DataLoader(train, batch_size=6, shuffle=False)
test_loader = data_utils.DataLoader(test, batch_size=1, shuffle=False)

# Set device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device} device")

#Model- and Hyperparameters
input_size=14
hidden_size=24
num_layers=6
num_classes=2
batch_size=8
sequence_length=6

# Initialize network 
model = RNN_Predictor(input_size, hidden_size, num_layers, num_classes).to(device)
print(model)

#Train

torch.manual_seed(0)
num_epochs=100
learning_rate=0.0001

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_list=[]
size = len(train_loader.dataset)

model.train()

# Train Model
for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader): 
        # Get data to cuda if possible
        x = x.to(device=device).squeeze(1)
        y = y.to(device=device)

        # forward
        scores = model(x)
        loss = criterion(scores, y)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent update step/adam step
        optimizer.step()
        loss_list.append(loss.item())

#Plot Loss Curve
plt.plot(loss_list)
plt.title('Decrease of Loss over Backpropagation Iteration')
plt.xlabel('Mini Batch Iteration')
plt.ylabel('Loss')
plt.show()

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")

summary(model,(6,14))

predicted=[]
model.eval()
print('Evaluate Testing Set:\n========')
for i, item in enumerate(x_test):
	with torch.no_grad():
		outp = model(item.unsqueeze(0))
		predicted.append(np.argmax(outp.detach()))


target = y_test
preds = torch.tensor(predicted,dtype=torch.long)
fa_scor=f1_score(preds, target, num_classes=2)
print(f"F1-Score =  {fa_scor:.2f}")

print(f"Predicted Class: {preds}")
print(f"Actual Calss:  {target}")

# #save model
# torch.save({
#     'epoch': epoch,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': loss,}, "..\model\RNN_Predictor.pt")

m = RNN_Predictor(input_size, hidden_size, num_layers, num_classes)
#print(m.state_dict())

torch.save(m.state_dict(), '.\model\RNN_Predictor.pt')

    # load model
m_state_dict = torch.load('.\model\RNN_Predictor.pt')
new_m = RNN_Predictor(input_size, hidden_size, num_layers, num_classes)
new_m.load_state_dict(m_state_dict)
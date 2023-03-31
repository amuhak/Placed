import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset
df = pd.read_csv("collegePlace.csv", header=None)
df.columns = ["Age","Gender","Stream","Internships","CGPA","Hostel","HistoryOfBacklogs","PlacedOrNot"]

# Encode the categorical variable
le = LabelEncoder()
df['Stream'] = le.fit_transform(df['Stream'])

# Standardize the numerical features
#scaler = StandardScaler()
#df[["Age","CGPA","Internships"]] = scaler.fit_transform(df[["Age","CGPA","Internships"]])

# Prepare the data
class PlacedDataset(Dataset):
    def __init__(self, data):
        self.X = data.iloc[:, :-1].values
        self.y = data.iloc[:, -1].values
        self.X = torch.tensor(self.X).float()
        self.y = torch.tensor(self.y).long()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Split into training and testing data
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

train_dataset = PlacedDataset(train_data)
test_dataset = PlacedDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Define the model
class PlacedModel(nn.Module):
    def __init__(self):
        super(PlacedModel, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1024)
        self.fc4 = nn.Linear(1024, 128)
        self.fc5 = nn.Linear(128, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = PlacedModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
n_epochs = int(input("Enter number of epochs: "))

for epoch in range(n_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader)}")
model.eval()
while True:
    print("ENTER ONLY INT VALUES")
    Age=int(input("enter age (int value) "))
    Gender=int(input("enter gender (int value) ")) #0==male,1==female
    Stream=int(input("enter stream (int vaue) "))
    Internships=int(input("enter internships (int vaue) "))
    CGPA=int(input("enter CGPA(int vaue) "))
    Hostel=int(input("enter Hostel(int vaue) "))
    HistoryOfBacklogs=int(input("enter if HistoryOfBacklogs(int vaue) "))
    x=torch.tensor([Age,Gender,Stream,Internships,CGPA,Hostel,HistoryOfBacklogs]).float().to(device)
    with torch.no_grad():
        prediction = model(x)
    print(prediction.shape)
    if prediction[0]>prediction[1]:
        print("YES")
    elif prediction[1]>prediction[0]:
        print("NO")
    else:
        print("IDK")
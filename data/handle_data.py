from torch.nn.utils.rnn import pad_sequence

merged_df = []
merged_df.extend(df)
merged_df.extend(rnd_df)
target = [1] * 10000
target.extend([0] * 10000)
print(len(merged_df))
print(len(target))
x = list(map(torch.tensor,merged_df))
print(len(max(x,key=len)))
print(len(pad_sequence(x, padding_value=-10.0, batch_first=True)[30]))
padded_df = pad_sequence(x, padding_value=10.0, batch_first=True)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        
    def __getitem__(self, index):
        return (self.x[index], self.y[index])

    def __len__(self):
        return len(self.x)

features_train, features_test, targets_train, targets_test = train_test_split(padded_df,
                                                                             target,
                                                                             test_size = 0.5,
                                                                             random_state = 42) 

# Wait, is this a CPU tensor now? Why? Where is .to(device)?
x_train_tensor = torch.from_numpy(np.array(features_train)).float()
y_train_tensor = torch.from_numpy(np.array(targets_train)).float()

x_test_tensor = torch.from_numpy(np.array(features_test)).float()
y_test_tensor = torch.from_numpy(np.array(targets_test)).float()

train_data = CustomDataset(x_train_tensor, y_train_tensor)
print(train_data[0])

test_data = TensorDataset(x_test_tensor, y_test_tensor)
print(x_test_tensor[0])

#should i suffle?
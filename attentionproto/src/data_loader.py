
class DataLoader:
    def __init__(self, data, batch_size=200, shuffle=True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
    

    def __len__(self):
        # Returns the number of batches
        return int(np.ceil(len(self.data) / self.batch_size))

    def __iter__(self):
        # Shuffles the indexes if required
        data = pd.DataFrame(self.data).to_numpy()
        data_size = len(data)
        num_batches_per_epoch = int((len(data)-1)/self.batch_size) + 1
      
        if self.shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * self.batch_size
            end_index = min((batch_num + 1) * self.batch_size, data_size)
            output = list(zip(*shuffled_data[start_index:end_index]))
            yield output[0],  output[1],  output[2],  output[3]
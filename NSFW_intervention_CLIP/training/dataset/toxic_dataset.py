from torch.utils.data import Dataset

def load_txt(file_path) :
    with open(file_path, 'r') as file :
        data = file.readlines()
    return data


class ToxicDataset(Dataset):
    def __init__(self, data_folder="./train/") :

        # Loading the three corresponding datasets
        toxic_words = load_txt(data_folder + "toxic_words.txt")
        toxic_mappings = load_txt(data_folder + 'toxic_mappings.txt')
        harmless_words = load_txt(data_folder + 'harmless_words.txt')

        print(len(toxic_words), len(toxic_mappings), len(harmless_words))
        # assert len(toxic_words) == len(toxic_mappings) == len(harmless_words)
        self.data = []
        for i in range(len(toxic_words)) :
            self.data.append({'toxic_words' : toxic_words[i], 'toxic_mappings' : toxic_mappings[i], 'harmless_words' : harmless_words[i]})
            

    def __getitem__(self, index):
        element = self.data[index]
        nsfw, safe, harmless = element['toxic_words'], element['toxic_mappings'], element['harmless_words']

        return (nsfw, safe, harmless)


    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


    def __len__(self):
        return len(self.data)
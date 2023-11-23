import panda as pd
import cv2
import numpy as np

class SyntheticImagesDataset(Dataset):
    def __init__(self, meta_data_csv_path: str, root_dir: str):
        """
        Arguments:
            meta_data_csv_path (string): Path to the csv file: rgb1_name, rgb2_name, depth1_name, depth2_name,
                                                     rot + translate 1 (12 numbers), rot + translate 2 (12 numbers)
            depth is 16 bit
            root_dir: Directory with all the images.
        """
        self.meta_data = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        images = [cv2.imread(os.path.join(self.root_dir, self.meta_data.iloc[idx, i])) for i in range(0, 4)]
        rot1 = np.array([self.meta_data.iloc[idx, i] for i in range(4, 13)])
        translate1 = np.array([self.meta_data.iloc[idx, i] for i in range(13, 16)])
        rot2 = np.array([self.meta_data.iloc[idx, i] for i in range(16, 25)])
        translate2 = np.array([self.meta_data.iloc[idx, i] for i in range(25, 28)])
        sample = {'rgb1': images[0], 'rgb2': images[1], 'depth1': images[2], 'depth2': images[3],
                  'rot1': rot1, 'translate1': translate1, 'rot2': rot2, 'translate2': transate2}
        return sample

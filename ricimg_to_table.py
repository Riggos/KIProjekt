import os
import pandas as pd


class imgorga:

    def __init__(self,location) -> None:
        self.origin = location
        self.labels = os.listdir(self.origin)
        self.label_paths = []
        self.image_paths = []
        self.numericdata_list = []
        self.feature_names = None
        self.panda_frame = None

        for xx in self.labels:
            self.label_paths.append(os.path.join(location, xx))

        for mypath in self.label_paths:
            imgs = [os.path.join(mypath, f) for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]
            self.image_paths.append(imgs)

    def get_feature_names(self,features_names):
        self.panda_frame = pd.DataFrame (self.numericdata_list, columns = features_names)
        return self.panda_frame

    def collect_numeric_data(self,feature_values):
        self.numericdata_list.append(feature_values)
        return 1
        

    def hardcode_makadata(self):
        return os.getcwd()


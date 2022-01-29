"""Dieses Python File beinhaltet eine Klasse "imgorga" um die Speicherorte und Labels der verschiedenen Bilder auszulesen,
um die Bilder einlesen zu können.
Zudem sammelt es extrahierte Features in eine Liste aus Vektoren (Ein Vektor gehört zu einem Bild) und wandelt diese 
in ein Panda.Dataframe um, sodass dieses leicht weiterverarbeitet werden kann.
"""

__author__ = "Eric Hirsch und Jonas Morsch"
__version__ = "1.0.0"

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
        """Gesammelte Vekotren als Panda_Frame ausgeben für die Weiterverarbeitung"""
        self.panda_frame = pd.DataFrame (self.numericdata_list, columns = features_names)
        return self.panda_frame

    def collect_numeric_data(self,feature_values):
        """Features eines Bildes als Vektor in eine Liste einschreiben"""
        self.numericdata_list.append(feature_values)
        return 1
        

    def hardcode_makadata(self):
        """Was ist der Standard-Pfad der Klasse"""
        return os.getcwd()


import numpy as np
import binoculars
from pathlib import Path

from matplotlib import pyplot as plt
from tqdm import tqdm


class FitaidOutput:
    """Base class for data output by BINoculars fitaid.
    """
    def __init__(self, file_path, type_and_scan_nr=None):
        self.file_path = file_path
        if not self.file_path.is_file():
            raise RuntimeError(f"File {self.file_path} does mot exist.")

        if type_and_scan_nr is None:
            # read from file name
            # files names always follow pattern "$type_mapped_scan_$scan#_$seq#.txt"
            if '_mapped_scan_' not in self.file_path.name:
                raise ValueError("Could not determine metadata from filename "
                                 f"for file {self.file_path}.")
            name_parts = self.file_path.name.split('_')
            self.type = name_parts[0]
            self.scan_nr = name_parts[4]
            self.seq_nr = name_parts[5]  # should not be needed for anything
        else:
            self.type, self.scan_nr = type_and_scan_nr

        # load data
        self.l_values, self.values = self.read_data()

    def read_data(self):
        """Read the data from the file.

        Fitaid output files are stored as simple text files with two colums
        without header. Colums are spearated by a space and contain floating
        point numbers in scientific notation.
        This format is simple enough to just use numpy for reading in.
        
        The first colum is L values, the second colum the corresponding values.
        """
        data = np.loadtxt(self.file_path)
        return data[:,0], data[:,1]
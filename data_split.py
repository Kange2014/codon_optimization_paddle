import random
from optparse import OptionParser
import numpy as np

def read_lines(path):
    """
    path: String, file path.
    return a list of sequence.
    """
    seqs = []
    with open(path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line):
                seqs.append(line)
    return seqs

class DataSetCreate():
    """
    A class to process data for sentiment analysis task.
    """
    def __init__(self,
                 data_path, ratio):
        """
        data_path: string, traing and testing dataset path
        """
        self.data_path = data_path
        self.ratio = ratio

    def split_data(self):
        """
        In order to shuffle fully, there is no need to load all data if
        each file only contains one sample, it only needs to shuffle list
        of file name. But one file contains multi lines, each line is one
        sample. It needs to read all data into memory to shuffle fully.
        This interface is mainly for data containning multi lines in each
        file, which consumes more memory if there is a great mount of data.

        data: the Dataset object to process.

        """
        data_list = []

        # read all data
        data_list = read_lines(self.data_path)

        length = len(data_list)
        random.shuffle(data_list)

        ratio = self.ratio
        training_num = int(length*ratio)

        training_data = data_list[0:training_num]
        testing_data = data_list[training_num:length]
        
        self.save_file(training_data,"training.txt")
        self.save_file(testing_data,"testing.txt")

    def save_file(self, data_list, filename):
        """
        Save data into file.
        data_list: a list of sequnece.
        filename: output file name.
        """
        f = open(filename, 'w')
        print "saving file: %s" % filename
        for seq in data_list:
            f.write('%s\n' % seq)
        f.close()

def option_parser():
    parser = OptionParser(usage="usage: python data_split.py "\
                                "-i data_dir [options]")
    parser.add_option(
        "-i",
        "--data",
        action="store",
        dest="input",
        help="Input data directory.")
    parser.add_option(
        "-r",
        "--ratio",
        action="store",
        type="float",
        dest="ratio",
        help="Ratio of splitting training and testing samples")

    return parser.parse_args()

def main():
    options, args = option_parser()
    data_dir = options.input
    ratio = options.ratio
    data_creator = DataSetCreate(data_dir,ratio)
    data_creator.split_data()


if __name__ == '__main__':
    main()
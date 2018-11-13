###############################################################################

from os import mkdir
from os.path import isdir



def make_new_dir(folder):
	if isdir(folder) == False:
		mkdir(folder)

def save_to_csv(dataframe, filepath, index_label):
	dataframe.to_csv(filepath, index_label=index_label, encoding="utf-8")
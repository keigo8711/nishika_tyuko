####################################################################
    # Combine files, Comvert category variables, Compensate null
####################################################################

import numpy as np
import pandas as pd
from MyPackege import cleaning


if __name__ == '__main__':
    path_to_get = '/Users/keigo/github/nishika/tyuko_mansyon/data/train'
    path_to_save = '/Users/keigo/github/nishika/tyuko_mansyon/data'
    Preprocessor = cleaning.Preprocessor(before_path=path_to_get, after_path=path_to_save)
    arranged_df = Preprocessor.return_df()
    # Preprocessor.save_df()
    # arranged_df = Preprocessor.recall_df()
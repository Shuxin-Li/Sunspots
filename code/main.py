import os
import numpy as np
from Boosting_CJ import Boosting as B_CJ
from Boosting import Boosting
import keras as K


if __name__ == '__main__':
    b_cj = B_CJ()
    b_zxb = Boosting()
    cj_label = b_cj.run()
    b_zxb.run(cj_label)
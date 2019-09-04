
# import sys
# sys.path.append("..")

# from models.base_models import BaseModel
# from .. import models

# from dynamic_sparse.models import *


import sys
import os
sys.path.append("../../")
# sys.path.append(os.path.expanduser("~/nta/nupic.research/projects/"))

import dynamic_sparse.models as models
import dynamic_sparse.networks as networks
from dynamic_sparse.common import *

print("test")
print(__file__)
print("Current File Name : ",os.path.basename(__file__))
print("Current File Path : ",os.path.realpath(__file__))

# PYTHONPATH=~/nta/nupic.research/projects/ python mlp_heb.py
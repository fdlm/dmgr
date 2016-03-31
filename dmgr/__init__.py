from datasets import Dataset
from . import (datasources, iterators, files, preprocessing)

import pkg_resources
__version__ = pkg_resources.get_distribution('dmgr').version
del pkg_resources

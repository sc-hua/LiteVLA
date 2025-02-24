## Origins 
`configs/` and `tools/` are copied from https://github.com/open-mmlab/mmdetection: `version 3.3.0`


## Modifications
`tools/train.py#12-15` and `tools/test.py#17-20` are added with lines as follows:

 ```python
import sys
sys.path.append("../../")

import models
```

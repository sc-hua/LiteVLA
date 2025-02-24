## Origins 
`configs/` and `tools/` are copied from https://github.com/open-mmlab/mmsegmentation: `version 1.2.2`

## Modifications
`tools/train.py#13-16` and `tools/test.py#9-12` are added with lines as follows:

```python
import sys
sys.path.append("../../")

import models
```

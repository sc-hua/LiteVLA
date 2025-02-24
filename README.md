# LiteVLA


we use cuda=11.8, python=3.10 and torch=2.1.2+cu118
```bash
conda create -n litevla python=3.10 -y
conda activate litevla
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```


## For downstream
```bash
pip install -U openmim
mim install mmengine==0.10.1 mmcv==2.1.0 mmdet==3.3.0 mmsegmentation==1.2.2
pip install opencv-python-headless ftfy regex
```
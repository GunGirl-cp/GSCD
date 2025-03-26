
# GSCD
 
## Installation
```
git clone https://github.com/GunGirl-cp/GSCD.git
cd GSCD
conda create -n gscd python=3.7
conda activate gscd
pip install -r requirements.txt
python setup.py
```

## Coarse change mask generation
```
bash ccmg.sh
```
This command will obtain the render images with change markers, the results are saved to `./logs/chess1/results_path_199999`


## Change detection refinement
We design an iterative optimization process which use CDR to refine inaccurate segmentation results.
```
python cdr.py --mask_dir <mask_path>  --img_dir <img_path> --output_dir <output_dir>
```

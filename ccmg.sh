python train.py -s data/chess --port 6017 --expname "test/chess" --configs arguments/llff/bikes.py 
python render.py --before_folder "output/test/chess"  --configs arguments/llff/bikes.py 
python cd.py --folder1 /home/honor410/Disk4T/lpf/4DGaussians/output/test/chess/renderresult/ours_60000/img1 --after_folder /home/honor410/Disk4T/lpf/4DGaussians/output/test/chess/renderresult/ours_60000/img1 --output_folder /home/honor410/Disk4T/lpf/4DGaussians/output/test/chess/renderresult/ours_60000/cd
python seg.py --mask_dir /home/honor410/Disk4T/lpf/4DGaussians/output/test/chess/renderresult/ours_60000/cd  --image_dir /home/honor410/Disk4T/lpf/4DGaussians/output/test/chess/renderresult/ours_60000/img1 --output_dir /home/honor410/Disk4T/lpf/GSCD/output/chess/mask1

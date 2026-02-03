# Out-of-Distribution Segmentation via Wasserstein-Based Evidential Uncertainty 

Here you can find the code associated with the paper named above (arxiv: https://arxiv.org/abs/2512.11373). You will have to set up MMsegmentation first.
This guide may help you:
https://mmsegmentation.readthedocs.io/en/latest/get_started.html

The setup can be a little bit tricky due to incompabilities between versions, here you can find what worked for me:

pip install torch==2.0.0 torchvision==0.15.1
pip install -U openmim
pip install mmengine
pip install mmcv
git clone -b main https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e 
python demo/image_demo.py demo/demo.png configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth --device cuda:0 --out-file result.jpg
pip install ftfy 				–no-cache-dir
pip install regex 				–no-cache-dir
pip install "numpy<2"


Besides the provided files, I made the following modifications to the standard setup:

For visuals regarding training I used WandB so i made these changes for mmsegmentation/tools/train.py:

line 6, 	Import wandb (importing weights and biases to access training data)
		line 91 + 92, 	#increase batch size to 8
    				cfg.train_dataloader.batch_size = 8 (overwrites the batch size of the cfg file and 									sets the Batch size to 8)
		line 96 - 106,   #WandB connection
    				wandb.init(config=cfg)
    				cfg.vis_backends =[dict(type='LocalVisBackend'),
              			dict(type='WandbVisBackend')]
            cfg.visualizer.vis_backends =[dict(type='LocalVisBackend'),
              			dict(type='WandbVisBackend')]     

In addition I modified the schedule, I used ADAM instead of standard SGD and i used a WarmUp of 1.500 iterations. How this works is explained in detail within the mmsegmentation documentation.




---

## Usage of the Wasserstein Metric
The main idea is to use a loss function based on the Wasserstein metric to encapture OOD objects in training. It has been seen that this preserves the underlying geometry within the probability simplex better than say a MSE loss.




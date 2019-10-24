### FILES IN THIS FOLDER

> Python scripts 

	1. class2color.ipynb 
		- Notebook that visualizes xview classes to unique color palletes 

	2. pycocoDemo.ipynb
		- Modified version of original pycocoDemo which also includes class2color visualization example on bottom of notebook


	In order to follow along with the scripts, you will need the data I used which were imported from the machine: 
	
		new_seg_imgs: /home/franklin/Documents/segmentron-master/new_seg_imgs

		make a new folder called: new_seg_xview
			- place in the validation images from /home/data/Xview/seg_chipped_500/  
			- place instance validation json from /home/franklin/Documents/segmentron-master/scripts
		
			- file structure of new_seg_xview should be: 
				> new_seg_xview
					> val2014 
					> instances_val2014.json 

### DOCUMENATION FOR TRAINING SCRIPTS ON MACHINE IN BIDS 

> SSH Information 

	Thought it might be convenient to provide user/pass to avoid permission issues if you ever do stuff in my folder:

	ssh franklin@169.229.63.109
	password: h4d123$$$

> faster-rcnn.pytorch

	> Info

		Location: /home/franklin/Documents/faster-rcnn.pytorch

		Data used to run script: 
			1. images 
				.../faster-rcnn.pytorch/data/coco/images
			2. annotations
				.../faster-rcnn.pytorch/data/coco/annotations

		For further details of folder/script, I recommend looking at [ https://github.com/jwyang/faster-rcnn.pytorch ]. 

	> Training: 

		Head to original folder location (/home/franklin/Documents/faster-rcnn.pytorch)

		Then Run: 
		CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset coco --net res101 --lr .0001 --bs 13 --lr_decay_gamma 1 --epochs 200 --use_tfb --cuda

	> Testing: 

		Head to original folder location (/home/franklin/Documents/faster-rcnn.pytorch)

		Then Run: 
		python test_net.py --dataset coco --net res101 --checksession 1 --checkepoch 100 --checkpoint 4237 --cuda


> segmentron-master

	> Info 

		Location: /home/franklin/Documents/segmentron-master

		Data used to run script: 
			1. images 
				.../segmentron-master/data/coco2014/val2014 
				.../segmentron-master/data/coco2014/train2014
			2. segmentation annotations: 
				.../segmentron-master/data/coco2014/annotations/seg_train2014 
				.../segmentron-master/data/coco2014/annotations/seg_val2014
				***(Instructions on how to produce these segmentation annotations is provided below)
			3. textfiles: 
				.../segmentron-master/data/coco2014/val2014.txt
				.../segmentron-master/data/coco2014/train2014.txt


	> Converting coco anotations to segementation masks (getting segmentation annotations)

		Head over to: /home/franklin/Documents/segmentron-master/scripts

		Make sure that the instance annotations are in the folder as well (as it should be right now)

		Then run the script to get train/val segmentations: 
			python convert_coco.py /home/data/XView/XView_chipped_500/ 'val' './seg_val'
			python convert_coco.py /home/data/XView/XView_chipped_500/ 'train' './seg_train'

				first arg - directory to images 
				second arg - which type (val or train)
				third arg - output 

	> Training 

		Head over to original folder location: Location: /home/franklin/Documents/segmentron-master

		The run
			python train.py --arch simple --backbone res101 --head k1 --dataset coco --max-epoch 200 --split 'train' --preproc torch --gpu 0 "./results"


	> Testing 
		When running this script, you will also output the segmentation predictions. To see how the segmentation predictions are being saved: 

			1. vi evaluate.py 
			2. type ':set number'
				- lines 29:31 = loading original images and saving to 'og_images' folder
				- lines 30:40 = converting segmentation mask to array and then saving to 'new_seg_imgs' folder 

			Adjust these you wish for your new data 

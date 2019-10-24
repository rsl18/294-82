# Setup Instructions

Note, for all the .sh scripts, you might have to grant execute permissions before being able to run them. You can do so like this: 

```bash
chmod +777 script_name.sh
```

## Folder structure:
To mimic how the files are setup on the GPU server (to avoid duplicating the large datasets once for each team member), you should set up a folder structure like this:

### If on BiD server:

```
home/
    data/
    experiments/
    <user_x>/
    	..h4d  <- this is this git repo
```
The h4d repo can be nested anywhere inside a user folder

### If on local machine:
```
<h4dmain>/
    data/
    experiments/
    h4d/  <- this is this git repo
```
`<h4dmain>/` can be named anything you want, we never reference that folder name anywhere. The other folder names should match the above. 
This git repo should be checked out into `<h4dmain>/h4d/`. If desired you can navigate to some place on your hard drive and then run the following commands to set up the folder structure and clone the repo: 

```bash
mkdir h4dmain
mkdir h4dmain/data
mkdir h4dmain/experiments
cd h4dmain
git clone git@github.com:mlaico/h4d.git h4d
```

All further operations assume you are in `<h4dmain>/h4d/` folder.

## Download Data
This step isn't automated because downloading Xview requires a login. To get the Xview dataset:

- Create an account on http://xviewdataset.org/, and download the "Download Training Images (tgz)" and "Download Training Labels (tgz)" links, and extract them to: <h4dmain>/data/Xview/vanilla/
- Your <h4dmain>/data/Xview/vanilla/ folder should contain a xView_train.json file, and a train_images folder


### (Optional):
If you want to speed up the setup process, you can copy the MSCOCO datasets from someone else or from BiD machine. The setup scripts will skip downloading MSCOCO from the web (~20GB) if it sees the zip files in the below locations.

Your data directory should look like this before you run setup.sh:

```
<H4D_DATA_DIR>/
    Xview/
        vanilla/
	    train_images/
	        *.tif
	    xView_train.geojson
    coco/
        images/
	    train2014.zip
	    val2014.zip
	annotations_trainval2014.zip
```

## Install Anaconda

Install [Anaconda](https://www.anaconda.com/distribution/) (miniconda should also work). The setup process will create the conda environment(s). 


## Run Setup:

Run: 

```bash
./setup.sh
```

## Now you should have the following setup:

### Shared directories

```
*data*/
	Xview/
		vanilla/
		coco_vanilla/
		coco_chipped/
	coco/
		images/
			train2014/
			val2014/
		annotations/
*experiments*/
	base_models/
		faster_rcnn/
			resnet101_caffe.pth
		ssd/
			vgg16_reducedfc.pth
		yolo/
			darknets53.conv.74
			yolov3-tiny.weights
			yolov3.weights
	faster_rcnn/
	ssd/
	yolov3/
```

### H4d Repo:

```
h4d/
	dataset/ ->symlink-> *data*
	experiments/ ->symlink-> *experiments*
	faster-rcnn.pytorch/
		data/
			coco/
				images/ ->symlink-> *data*/Xview/coco_chipped
				annotations/ ->symlink-> *data*/Xview/coco_chipped
			pretrained_model/ ->symlink-> *experiments*/base_models/faster_rcnn/
		models/ ->symlink-> *experiments*/faster_rcnn/
	ssd-master/
		datasets/
			annotations/ 	(symlink)
			train2014/ 	(symlink)
			val2014/ 	(symlink)
	yolov3/
		data/
			coco/
				images ->symlink-> *data*/Xview/coco_chipped
				annotations ->symlink-> *data*/Xview/coco_chipped
	ssd-master/
	<scripts>
	...
```

## TODO:

- [ ] Add instructions on how to customize setup. e.g., if you want to setup multiple clones of the h4d repo side-by-side to run trainings from different versions of the code in parallel, you would run environment creation and data setups once, but other tasks like framework-specific setup scripts require to be run one for each clone of the repo. If not too much work, maybe automate that instead of documenting it.


# Setup Instructions

Note, for all the .sh scripts, you might have to grant execute permissions before being able to run them. You can do so like this: 

```bash
chmod +777 script_name.sh
```

## Folder structure:
To mimic how the files are setup on the GPU server (to avoid duplicating the large datasets once for each team member), you should set up a folder structure like this:
```
...
<main>/
    data/Xview/vanilla
    experiments/
    overhead/  <- this is this git repo
```
`<main>/` can be named anything you want, we never reference that folder name anywhere. The other folder names should match the above. 

## Download Data
This step isn't automated because downloading Xview requires a login. To get the Xview dataset:

- Create an account on http://xviewdataset.org/, and download the "Download Training Images (tgz)" and "Download Training Labels (tgz)" links, and extract them to: <h4dmain>/data/Xview/vanilla/

-IMPORTANT - the TGZ file is a lot easier to deal with. There is disclaimer on the download page about corruption errors from the zip file. You can follow that. Or you can just download the TGZ file.

- Your <main>/data/Xview/vanilla/ folder should contain a xView_train.json file, and a train_images folder

## Install Anaconda

Install [Anaconda](https://www.anaconda.com/distribution/) (miniconda should also work). The setup process will create the conda environment(s). 


## Run Setup:

Run in overhead folder

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
*experiments*/
overhead/
	dataset/ ->symlink-> *data*
	experiments/ ->symlink-> *experiments*
	centernet-master/
		data ->symlink *overhead/datasets* or *<main>/data*
		exp ->symlink *overhead/experiments* or *<main>/experiments*
	...
```
## To run centernet:

```
conda activate h4d_env
*edit and run one of the xview bash scripts in centernet-master/experiments*
```

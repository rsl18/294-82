from . import register_dataset
from ..seg import SegData


@register_dataset('coco')
class COCO(SegData):
    """
    Load semantic segmentation data from COCO converted to VOC style.

    Take
        root_dir: path to COCO year dir
        split: {train,val}
    """

    #TODO
 #    classes = ['__background__',
	# 'Fixed-wing Aircraft',
 #         'Small Aircraft',
 #         'Cargo Plane',
 #         'Helicopter',
 #         'Passenger Vehicle',
 #         'Small Car',
 #         'Bus',
 #         'Pickup Truck',
 #         'Utility Truck',
 #         'Truck',
 #         'Cargo Truck',
 #         'Truck w/Box',
 #         'Truck Tractor',
 #         'Trailer',
 #         'Truck w/Flatbed',
 #         'Truck w/Liquid',
 #         'Crane Truck',
 #         'Railway Vehicle',
 #         'Passenger Car',
 #         'Cargo Car',
 #         'Flat Car',
 #         'Tank car',
 #         'Locomotive',
 #         'Maritime Vessel',
 #         'Motorboat',
 #         'Sailboat',
 #         'Tugboat',
 #         'Barge',
 #         'Fishing Vessel',
 #         'Ferry',
 #         'Yacht',
 #         'Container Ship',
 #         'Oil Tanker',
 #         'Engineering Vehicle',
 #         'Tower crane',
 #         'Container Crane',
 #         'Reach Stacker',
 #         'Straddle Carrier',
 #         'Mobile Crane',
 #         'Dump Truck',
 #         'Haul Truck',
 #         'Scraper/Tractor',
 #         'Front loader/Bulldozer',
 #         'Excavator',
 #         'Cement Mixer',
 #         'Ground Grader',
 #         'Hut/Tent',
 #         'Shed',
 #         'Building',
 #         'Aircraft Hangar',
 #         'Damaged Building',
 #         'Facility',
 #         'Construction Site',
 #         'Vehicle Lot',
 #         'Helipad',
 #         'Storage Tank',
 #         'Shipping container lot',
 #         'Shipping Container',
 #         'Pylon',
 #         'Tower']

    # pixel statistics (RGB)
    mean = (0.48109378, 0.45752457, 0.40787054)
    std = (0.27363777, 0.26949592, 0.28480016)

    # reserved target value to exclude from loss, evaluation, ...
    ignore_index = 255

    def __init__(self, **kwargs):
        kwargs['root_dir'] = kwargs.get('root_dir', None) or 'data/coco'
        super().__init__(**kwargs)

    def load_slugs(self):
        with open(self.listing_path(), 'r') as f:
            slugs = f.read().splitlines()
        return slugs

    def listing_path(self):
        return self.root_dir / f'{self.split}.txt'

    def slug_to_image_path(self, slug):
        return self.root_dir / f'{self.split}' / f'{slug}'

    def slug_to_annotation_path(self, slug):
        return (self.root_dir / f'annotations/seg_{self.split}'
                / f'{slug}'.replace('jpeg', 'png').replace("jpg", "png"))

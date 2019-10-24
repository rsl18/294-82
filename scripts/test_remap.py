import csv
from pycocotools.coco import COCO
from pathlib import Path

"""
THE FOLLOWING VARIABLES NEED TO BE SPECIFIED BEFORE RUNNING TESTS:
	old_ds_annotations, mapped_json, and csv_file

IF TESTING WITH ALREADY REMAPPED DATASET:
	set new_coco_obj to COCO(mapped_json) where mapped_json = folder / [already_mapped.json]

IF CREATING NEW DATASET WITH REMAPPED CATEGORIES WHILE TESTING:
	set new_coco_obj to COCO(old_ds_annotations, True, csv_file, mapped_json)
	where mapped_json is the desired path for the output JSON file to be written
"""

vanilla_folder = Path('../datasets/Xview/coco_vanilla')
h4d_folder = Path('../')
old_ds_annotations = vanilla_folder / 'xview_coco_v0.json'
mapped_json = vanilla_folder / 'xview_coco_v2.json'
csv_file = h4d_folder / 'xview_v2.csv'

# Setting up old unmapped dataset and new mapped dataset for comparison testing
old_coco_obj = COCO(old_ds_annotations)
oset = old_coco_obj.dataset
new_coco_obj = COCO(old_ds_annotations, True, csv_file, mapped_json)
nset = new_coco_obj.dataset


# Getting all rows from the CSV (mapping scheme)
map_rows = []
with open(csv_file, newline='') as csvfile:
		spamreader = csv.reader(csvfile, quotechar='|')
		for row in spamreader:
			map_rows.append(row)
map_rows = map_rows[1:]


def check_values_same(lst1, lst2):
	"""
	Check if two lists have the same contents without checking order (works only for 1D array).
	"""
	return len(lst1) == len(lst2) and sorted(lst1) == sorted(lst2)

def test_mapping_switch():
	"""
	Ensure unmapped and mapped datasets are different (checking whether create_mapping was set to True or False).
	"""
	assert nset != oset

def test_NL_has_NA():
	"""
	Check if old categories and old annotations were deleted entirely for rows where NL/NP is 'NA'.
	"""
	old_anns = oset['annotations']
	anns_to_check = []
	cats_to_check = []
	for ann in old_anns[:]:
		ann_id = ann['id']
		cat_id = ann['category_id']
		cat = old_coco_obj.cats[cat_id]
		OL = cat['name']
		for search_row in map_rows:
			if OL == search_row[0]:
				row = search_row 
				OP, NL, NP = cat['supercategory'], row[2], row[3]

				# now we have the particular row from the CSV whose old category corresponds to this annotation's category

				if NL == 'NA':
					anns_to_check.append(str(ann_id))
					if cat not in cats_to_check:
						cats_to_check.append(cat_id)

	anns_in_new = [new_coco_obj.anns.get(ann, 'not found') for ann in anns_to_check]
	cats_in_new = [new_coco_obj.cats.get(cat, 'not found') for cat in cats_to_check]
	anns_in_old = [old_coco_obj.anns.get(ann, 'not found') for ann in anns_to_check]
	cats_in_old = [old_coco_obj.cats.get(cat, 'not found') for cat in cats_to_check]
	found_anns_new = [False if item == 'not found' else True for item in anns_in_new]
	found_cats_new = [False if item == 'not found' else True for item in cats_in_new]
	found_anns_old = [False if item == 'not found' else True for item in anns_in_old]
	found_cats_old = [False if item == 'not found' else True for item in cats_in_old]

	assert True not in found_anns_new and True not in found_cats_new
	assert False not in found_anns_old and False not in found_cats_old


def test_NL_NP_correct():
	"""
	Check that all new categories match label scheme in csv.
	"""
	NLNP_pairs = []
	for search_row in map_rows:
		NL, NP = search_row[2], search_row[3]
		if not NL == 'NA':
			NLNP_pairs.append((NL, NP))
	is_cat_correct = []
	for cat in nset['categories']:
		cat_pair = (cat['name'], cat['supercategory'])
		is_cat_correct.append(cat_pair in NLNP_pairs)

	assert False not in is_cat_correct


def test_annotations_mapping():
	"""
	Check that all annotations have been properly mapped to the new categories.
	"""
	old_pair_to_new_pair = {}
	for search_row in map_rows:
		OL, OP, NL, NP = search_row[0], search_row[1], search_row[2], search_row[3]
		if not NL == 'NA':
			old_pair_to_new_pair[(OL, OP)] = (NL, NP)

	# rows where multiple old categories map to a single new category
	surjections = []
	# rows where only one old category maps to a single new category
	one_to_ones = []
	for name_parent_pair in old_pair_to_new_pair:
		if len(old_pair_to_new_pair[name_parent_pair]) > 1:
			surjections.append(name_parent_pair)
		else:
			one_to_ones.append(name_parent_pair)

	
	is_ann_correct = []
	for OL_OP_pair in surjections + one_to_ones:
		for old_cat in oset['categories']:
			if old_cat['name'] == OL_OP_pair[0] and old_cat['supercategory'] == OL_OP_pair[1]:
				old_cat_id = old_cat['id']
				anns_to_check = []
				for ann in oset['annotations']:
					if ann['category_id'] == old_cat_id:
						anns_to_check.append(ann)
				
				# now we have all the annotations associated with this particular old category
				NL, NP = old_pair_to_new_pair[(OL_OP_pair[0], OL_OP_pair[1])]
				for new_cat in nset['categories']:
					if new_cat['name'] == NL and new_cat['supercategory'] == NP:
						for ann in anns_to_check:
							ann_id = ann['id']
							new_ann = new_coco_obj.anns[ann_id]
							is_ann_correct += [new_ann['category_id'] == new_cat['id']]

	assert False not in is_ann_correct

def test_no_missing_cats():
	"""
	Check that all new categories in the CSV exist
	"""
	rows_found = []
	map_rows_no_NA = []
	for search_row in map_rows:
		NL, NP = search_row[2], search_row[3]
		if not NL == 'NA':
			map_rows_no_NA.append(search_row)
	for search_row in map_rows_no_NA:
		NL, NP = search_row[2], search_row[3]
		for cat in nset['categories']:
			if cat['name'] == NL and cat['supercategory'] == NP:
				rows_found.append(search_row)

	rows_not_found = [row for row in map_rows_no_NA if row not in rows_found]

	assert not rows_not_found


def test_no_extra_ids():
	"""
	Check that there exists only one id for each new category in the csv and that all obsolete ids were deleted
	"""
	NLNP_pairs = set()
	all_ids = set()
	for cat in nset['categories']:
		all_ids.add(cat['id'])
	
	for search_row in map_rows:
		NL, NP = search_row[2], search_row[3]
		if not NL == 'NA':
			NLNP_pairs.add((NL, NP))

	assert len(all_ids) == len(nset['categories'])
	assert len(NLNP_pairs) == len(all_ids)

def test_total_new_annotations():
	"""
	Check that all old annotations exist except for annotations that were deleted due to old category being mapped to 'NA'
	"""
	old_num_anns = len(oset['annotations'])
	new_num_anns = len(nset['annotations'])
	num_NAs_found = 0

	old_anns = oset['annotations']
	for ann in old_anns:
		ann_id = ann['id']
		cat_id = ann['category_id']
		cat = old_coco_obj.cats[cat_id]
		OL = cat['name']
		for search_row in map_rows:
			if OL == search_row[0]:
				row = search_row 
				NL = row[2]

				# now we have the particular row from the CSV whose old category corresponds to this annotation's category
				if NL == 'NA':
					num_NAs_found += 1

	assert old_num_anns - num_NAs_found == new_num_anns

def test_sequential_cat_IDs():
	"""
	Check that all category IDs are unique and sequential, beginning at index 0.
	"""

	new_cat_ids = sorted(list(set([cat['id'] for cat in nset['categories']])))
	assert new_cat_ids == list(range(len(new_cat_ids)))

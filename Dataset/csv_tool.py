import csv
import os


def build_csv(root, filename_flk, filename_flkfree):
    '''
     root: str, path of the data to be synthesized
     filename_flk : str, csv filename of flickering images 
     filename_flkfree : str, csv filename of flicker-free images 
    '''
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):
        if not os.path.isdir(os.path.join(root, name)):
            continue
        name2label[name] = len(name2label.keys())
 
    ENF, nonENF = [],[]
    for name in name2label.keys():
        images = []
        i = 0
        j = 0
        images += glob.glob(os.path.join(root, name, '*.png'))
        images += glob.glob(os.path.join(root, name, '*.jpg'))
        images += glob.glob(os.path.join(root, name, '*.jpeg'))
        
        # root data are divided in two parts; 
        # [0-m] are flickering images, [m:m+l] are flicker-free images,
        
        for m in range(50):
            ENF.append(images[i])
            i += 1
        
        for l in range(50):
            nonENF.append(images[j+50])
            j += 1

    with open(os.path.join(root, filename_flk), mode='w', newline='') as f:
        writer = csv.writer(f)
        for img in ENF:
            writer.writerow([img])
        print('written into csv file:', filename_flk)
        
    with open(os.path.join(root, filename_flkfree), mode='w', newline='') as f:
        writer = csv.writer(f)
        for img in nonENF:
            writer.writerow([img])
        print('written into csv file:', filename_flkfree)     
        
        
def build_csv_single(root, filename_flk):
    name2label = {}
    for name in sorted(os.listdir(os.path.join(root))):

        if not os.path.isdir(os.path.join(root, name)):
            continue
        name2label[name] = len(name2label.keys())

    ENF = []
    for name in name2label.keys():
        images = []
        i = 0
        images += glob.glob(os.path.join(root, name, '*.png'))
        images += glob.glob(os.path.join(root, name, '*.jpg'))
        images += glob.glob(os.path.join(root, name, '*.jpeg'))
        for m in images:
            ENF.append(m)
            
    with open(os.path.join(root, filename_flk), mode='w', newline='') as f:
        writer = csv.writer(f)
        for img in ENF:

            writer.writerow([img])
        print('written into csv file:', filename_flk)
        
        
def load_csv(root, filename):
    images = []
    with open(os.path.join(root, filename)) as f:
        reader = csv.reader(f)
        for row in reader:
            img = row            
            images.append(img)

    return images

def getnewList(newlist):
	d = []
	for element in newlist:
		if not isinstance(element,list):
			d.append(element)
		else:
			d.extend(getnewList(element))
	
	return d
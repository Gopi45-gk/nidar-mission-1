import os
import glob
import xml.etree.ElementTree as ET

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(xml_path, output_path):
    try:
        in_file = open(xml_path)
        tree = ET.parse(in_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        
        # If width or height is 0, we can't normalize. Skip or warn.
        if w == 0 or h == 0:
            print(f"Warning: Zero size in {xml_path}. Skipping.")
            return

        out_file = open(output_path, 'w')
        
        for obj in root.iter('object'):
            difficult = obj.find('difficult')
            if difficult is not None:
                difficult = difficult.text
                if int(difficult) == 1:
                    continue
            
            cls = obj.find('name').text
            # Identify class ID. Assuming class names are already integers as seen in sample.
            # If not, you would need a mapping dictionary here.
            try:
                cls_id = int(cls)
            except ValueError:
                print(f"Warning: Non-integer class name '{cls}' found in {xml_path}. Skipping object.")
                continue
                
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
            bb = convert((w,h), b)
            out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            
        in_file.close()
        out_file.close()
    except Exception as e:
        print(f"Error converting {xml_path}: {e}")

if __name__ == '__main__':
    work_dir = os.path.dirname(os.path.abspath(__file__))
    anno_dir = os.path.join(work_dir, 'anno')
    labels_dir = os.path.join(work_dir, 'labels')
    
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
        
    xml_files = glob.glob(os.path.join(anno_dir, '*.xml'))
    print(f"Found {len(xml_files)} xml files.")
    
    for xml_file in xml_files:
        basename = os.path.basename(xml_file)
        out_path = os.path.join(labels_dir, basename.replace('.xml', '.txt'))
        convert_annotation(xml_file, out_path)
        
    print("Conversion complete.")

import os
import xml.etree.ElementTree as ET
import csv


def xml_to_csv(xml_dir, output_dir):
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith('.xml'):
            tree = ET.parse(os.path.join(xml_dir, xml_file))
            root = tree.getroot()

            csv_file = os.path.splitext(xml_file)[0] + '.csv'
            csv_path = os.path.join(output_dir, csv_file)

            with open(csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])

                for obj in root.findall('object'):
                    filename = os.path.join(xml_dir, xml_file.replace('.xml', '.jpg'))
                    width = root.find('size/width').text
                    height = root.find('size/height').text
                    class_name = obj.find('name').text
                    xmin = obj.find('bndbox/xmin').text
                    ymin = obj.find('bndbox/ymin').text
                    xmax = obj.find('bndbox/xmax').text
                    ymax = obj.find('bndbox/ymax').text

                    writer.writerow([filename, width, height, class_name, xmin, ymin, xmax, ymax])


# Usage example
xml_directory = ""

output_directory = ""
xml_to_csv(xml_directory, output_directory)

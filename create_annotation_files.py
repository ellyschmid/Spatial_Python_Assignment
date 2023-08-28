import os
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

import xml.etree.ElementTree as ET
from xml.dom import minidom

# Callback function for rectangle selection
def line_select_callback(clk, rls):
    global tl_list
    global br_list
    tl_list.append((int(clk.xdata), int(clk.ydata)))  # Top-left point
    br_list.append((int(rls.xdata), int(rls.ydata)))  # Bottom-right point

# Toggle rectangle selector
def toggle_selector(event):
    toggle_selector.RS.set_active(True)

# Key press event handler
def onkeypress(event):
    global tl_list
    global br_list
    if event.key == 'q':
        generate_xml(tl_list, br_list, file_name, name_class)
        tl_list = []
        br_list = []

# Function to generate XML annotation
def generate_xml(tl_list, br_list, file_name, name_class):
    annotation = ET.Element("annotation")

    filename = ET.SubElement(annotation, "filename")
    filename.text = file_name

    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    height = ET.SubElement(size, "height")
    depth = ET.SubElement(size, "depth")
    width.text = str(abs(tl_list[0][0] - br_list[0][0]))
    height.text = str(abs(tl_list[0][1] - br_list[0][1]))
    depth.text = "3"  # Assuming RGB images

    object_elem = ET.SubElement(annotation, "object")
    name_elem = ET.SubElement(object_elem, "name")
    name_elem.text = name_class

    bndbox = ET.SubElement(object_elem, "bndbox")
    xmin = ET.SubElement(bndbox, "xmin")
    ymin = ET.SubElement(bndbox, "ymin")
    xmax = ET.SubElement(bndbox, "xmax")
    ymax = ET.SubElement(bndbox, "ymax")
    xmin.text = str(tl_list[0][0])
    ymin.text = str(tl_list[0][1])
    xmax.text = str(br_list[0][0])
    ymax.text = str(br_list[0][1])

    xml_str = ET.tostring(annotation, encoding="unicode")
    dom = minidom.parseString(xml_str)
    pretty_xml_str = dom.toprettyxml()

    xml_filename = os.path.join('path where annotations should be saved', file_name.replace('.png', '.xml'))
    with open(xml_filename, "w") as xml_file:
        xml_file.write(pretty_xml_str)

# Main
image_folder = 'Path to Image folder'
file_name = ''
name_class = ''

tl_list = []
br_list = []
file_names = os.listdir(image_folder)

for file_name in file_names:
    if file_name[0] != '.':
        name_class, sep, tail = file_name.partition('_')
        dir_file = os.path.join(image_folder, file_name)

        fig, ax = plt.subplots(1)
        image = cv2.imread(dir_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(image)

        toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )

        bbox = plt.connect('key_press_event', toggle_selector)
        key = plt.connect('key_press_event', onkeypress)
        plt.show()

print('Number of Processed Images:', len(file_names))

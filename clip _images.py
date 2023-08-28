
import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask


def clip_image_with_shapefile(input_image, shapefile, output_folder):
    # Read the shapefile
    gdf = gpd.read_file(shapefile)

    # Open the input image
    with rasterio.open(input_image) as src:
        for idx, geom in enumerate(gdf.geometry):
            # Clip the image with the polygon
            out_image, out_transform = mask(src, [geom], crop=True)

            # Copy the metadata
            out_meta = src.meta.copy()

            # Update metadata with new dimensions and transform
            out_meta.update({"driver": "PNG",  # Set the output format to JPEG
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})

            # Define the output filename with a .jpg extension
            output_filename = os.path.join(output_folder, f'ship_{idx}.png')

            # Save the clipped image as JPEG
            with rasterio.open(output_filename, "w", **out_meta) as dest:
                dest.write(out_image)

if __name__ == "__main__":
    input_image = ''  # Replace with your input image file (JPEG, PNG..)
    shapefile = ''  # Replace with your input shapefile
    output_folder = ''  # Folder to save the clipped images

    os.makedirs(output_folder, exist_ok=True)

    clip_image_with_shapefile(input_image, shapefile, output_folder)

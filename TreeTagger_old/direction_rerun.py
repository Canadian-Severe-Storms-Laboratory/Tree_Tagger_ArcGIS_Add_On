import numpy as np
np.seterr(all="ignore")
from treetag import *
import sys
import os
import timeit
from datetime import datetime


# timer to measure total runtime
t = timeit.default_timer()

#list of arguments passed from arcgis add in
argv = []
# path to arcgis project folder
project_path = ""

print("Loading arguments")
try:
    #geting project path
    argv = sys.argv[1:]
    project_path = argv[0]
except IndexError:
    print_error("Couldn't load Tree Tagger settings, make sure the entered settings are correct")

try:
    argv = load_args(project_path + "\\TreeTagger\\args.txt")
except FileNotFoundError:
    print_error("Couldn't load Tree Tagger arguments file, make sure the entered settings are correct")

#vector shapefile path
vector_file_path = argv[0].replace("%20", " ")

#should the direction shape files be combined into a single file?
render_as_points = int(argv[1])

#should the direction shape files be combined into a single file?
combined_directions = int(argv[2])

#direction interpolation type
direction_interpolation = argv[3]

#coordinate of top left corner
top_left = (float(argv[4]), float(argv[5]))

#coordinate of top left corner
bottom_right = (float(argv[6]), float(argv[7]))

#imagery scale
image_scale = (float(argv[8]), float(argv[9]))

#grid sizes for averaging direction vectors
direction_grid_parameters = [int(x) for x in argv[10:]]
direction_grid_parameters = list(np.array_split(direction_grid_parameters, len(direction_grid_parameters)/2))

vectors = []

try:
    vectors = load_vector_shapefile(vector_file_path, image_scale, top_left)
except Exception:
    print_error("Failed to load shapefile: " + vector_file_path)

print("getting average directions")
grid_vectors = build_vectors(direction_grid_parameters, direction_interpolation, top_left, bottom_right, vectors, image_scale)

print("Exporting directions to shape files")
project_pathExtension = "/TreeTagger/Direction_Rerun_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
try:
    os.mkdir(project_path + project_pathExtension)
    file_path = project_path + project_pathExtension + '/Directions_'

    if combined_directions == 0:
        export_direction_shapefile(grid_vectors, file_path, direction_grid_parameters, direction_interpolation, using_points=render_as_points)
    else:
        export_combined_direction_shapefile(grid_vectors, file_path, direction_grid_parameters, direction_interpolation, using_points=render_as_points)

except Exception:
    print_error("couldn't create direction shapefile, make sure TreeTagger folder exists in your ArcGIS project directory")

#program has finished, print total run time in seconds
print("time: ", timeit.default_timer() - t)


























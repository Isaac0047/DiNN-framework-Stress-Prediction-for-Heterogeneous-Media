## Description

This folder contains the MATLAB code to generate Plate with circular cutout models with spatial randomness (PC-SR).

##### To generate the PC-SR models, you will need to follow below steps:

(1) include Composite_fix_hole_hollow.m, Matlab2Abaqus_fix.m and Master_file_fix.inp in target directory

(2) make necessary changes inside Composite_fix_hole_hollow.m, for geometry shape, size and material assignment

(3) run Composite_fix_hole_hollow.m

##### To postprocess the ABAQUS solution .odb file, you will need to follow below steps:

(1) include Cartesian_Map.m, Composite_post_one_hole_hollow.m, distance.m and stress_interp_hollow.m in target directory

(2) make necessary chagnes inside Composite_post_one_hole_hollow.m for Cartesian Map size change and the Geometry contour labelling

(3) run Composite_post_one_hole_hollow.m

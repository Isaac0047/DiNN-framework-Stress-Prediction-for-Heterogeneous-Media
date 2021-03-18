## Description

This folder contains the MATLAB code to generate Square packed fiber reinforced composite models with uncontrolled spatial randomness (SP-SR-UC).

##### To generate the SP-SR-UC models, you will need to follow below steps:

(1) include Composite_center.m, Matlab2Abaqus_center.m and Master_file_fix.inp in target directory

(2) make necessary changes inside Composite_center.m, for geometry shape, size and material assignment

(3) run Composite_center.m

##### To postprocess the ABAQUS solution .odb file, you will need to follow below steps:

(1) include Cartesian_Map.m, Composite_center_small_post_random.m, distance.m and stress_interp_bi.m in target directory

(2) make necessary chagnes inside Composite_center_small_post_random.m for Cartesian Map size change and the Geometry contour labelling

(3) run Composite_center_small_post_random.m

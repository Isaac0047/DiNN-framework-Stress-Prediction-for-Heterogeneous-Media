## Description

This folder contains the MATLAB code to generate Square packed fiber reinforced composite models with volume fraction randomness (SP-VR).

##### To generate the SP-VR models, you will need to follow below steps:

(1) include Composite_center.m, Matlab2Abaqus_center.m and Master_file_fix.inp in target directory

(2) make necessary changes inside Composite_center.m, for geometry shape, size and material assignment

(3) run Composite_center.m

##### To postprocess the ABAQUS solution .odb file, you will need to follow below steps:

(1) include Cartesian_Map.m, Composite_center_small_post.m, distance.m and stress_interp_bi.m in target directory

(2) make necessary chagnes inside Composite_center_small_post.m for Cartesian Map size change and the Geometry contour labelling

(3) run Composite_center_small_post.m

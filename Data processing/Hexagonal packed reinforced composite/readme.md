## Description

This folder contains the MATLAB code to generate Hexagonal packed fiber reinforced composite models with volume fraction randomness (HP-VR).

##### To generate the SP-VR models, you will need to follow below steps:

(1) include Composite_hex.m, Matlab2Abaqus_hex.m and Master_file_fix.inp in target directory

(2) make necessary changes inside Composite_hex.m, for geometry shape, size and material assignment

(3) run Composite_hex.m

##### To postprocess the ABAQUS solution .odb file, you will need to follow below steps:

(1) include Cartesian_Map.m, Composite_hex_post.m, distance.m and stress_interp_hex.m in target directory

(2) make necessary chagnes inside Composite_hex_post.m for Cartesian Map size change and the Geometry contour labelling

(3) run Composite_hex_post.m

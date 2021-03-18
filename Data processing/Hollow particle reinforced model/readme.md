## Description

This folder contains the MATLAB code to generate Hollow particle reinforced composite models with volume fraction randomness (HPR-VR).

##### To generate the HPR-VR models, you will need to follow below steps:

(1) include Composite_hollow_ring.m, Matlab2Abaqus_hollow.m and Master_file_fix.inp in target directory

(2) make necessary changes inside Composite_hollow_ring.m, for geometry shape, size and material assignment

(3) run Composite_hex.m

##### To postprocess the ABAQUS solution .odb file, you will need to follow below steps:

(1) include Cartesian_Map.m, Composite_post_hollow_ring.m, distance.m and stress_interp_hollow.m in target directory

(2) make necessary chagnes inside Composite_post_hollow_ring.m for Cartesian Map size change and the Geometry contour labelling

(3) run Composite_post_hollow_ring.m

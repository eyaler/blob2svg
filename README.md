Convert a raster image containing blobs to svg.

Assumptions:

1. Single channel image
2. Blobs are defined by given levels (e.g. 0=background 1-255=blobs)
3. Blobs have genus-0 topologies (holes and enclosed shapes are ignored)
4. Output blobs are single color

Requirements: 

1. numpy
2. scipy
3. opencv
4. wand (optional, for saving a png copy)
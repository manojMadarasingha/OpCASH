# OpCASH
Code repository for the paper OpCASH: Optimized Utilization of MEC Cache for 360-Degree Video Streaming with Dynamic Tiling

This repository holds the artifacts to generate the main results of the OpCASH.

## Requirements
Following packages are required.

* Numpy				
*	Pandas			
*	matplotlib
*	scikit-image      0.16.2
*	glpk              0.4.6
*	networkx          2.4

## Run the script
To run the script clone the repository to your local repository and install the required packages above. Then run the command
`python3 opcash_main.py`

## How the code works
The input data (used in test purpose) is provided in the below folders

* Tile_info_DT  : Variable tile information of the users (for each video and each chunk). In a given `.csv` file, DT information is given as follows.
  * l_l_m: lower left y
  * l_l_n: lower left x
  * u_r_m: upper right y
  * u_r_n: upper right x
    









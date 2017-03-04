#!/bin/bash
# tsdf - tiny sdf generator
#   usage: ./tsdf.sh def_file sdf_file [vtk_file]
#
# TEST: tsdf1
# ./tsdf.sh examples/ywall1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf2
# ./tsdf.sh examples/ywall2.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf3
# ./tsdf.sh examples/sphere1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf4
# ./tsdf.sh examples/sphere2.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf5
# ./tsdf.sh examples/cylinder1.tsdf sdf.dat sdf.out.vti
#
# TEST: tsdf6
# ./tsdf.sh examples/cylinder2.tsdf sdf.dat sdf.out.vti

# A wrapper for sdf.awk
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
TSDF_CONFIG_DIR="$DIR" "$DIR"/tsdf.awk "$@"


$B/algo/edg/imp.o: $S/utils/error.h $S/algo/edg/imp.h $S/algo/edg/imp/main.h
$B/algo/kahan_sum/imp.o: $S/utils/imp.h $S/utils/error.h $S/algo/kahan_sum/imp.h $S/algo/kahan_sum/imp/main.h $S/algo/kahan_sum/imp/type.h
$B/algo/key_list/imp.o: $S/utils/imp.h $S/utils/error.h $S/algo/key_list/imp/util.h $S/algo/key_list/imp.h $S/utils/string/imp.h $S/algo/key_list/imp/main.h $S/algo/key_list/imp/type.h $S/utils/msg.h
$B/algo/scalars/imp.o: $S/utils/imp.h $S/utils/error.h $S/algo/scalars/imp.h $S/algo/vectors/imp.h $S/algo/scalars/imp/main.h $S/algo/scalars/imp/type.h
$B/algo/vectors/imp.o: $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/algo/vectors/imp.h $S/algo/vectors/imp/main.h $S/algo/vectors/imp/type.h $S/math/tform/imp.h $S/coords/imp.h
$B/conf/imp.o: $S/utils/imp.h $S/utils/error.h $S/conf/imp/set.h $S/conf/imp.h $S/conf/imp/main.h $S/conf/imp/type.h $S/conf/imp/get.h $S/utils/msg.h
$B/coords/conf.o: $S/utils/imp.h $S/coords/ini.h $S/utils/error.h $S/conf/imp.h
$B/coords/imp.o: $S/utils/imp.h $S/inc/conf.h $S/coords/ini.h $S/utils/error.h $S/mpi/wrapper.h $S/coords/imp.h $S/utils/mc.h $B/conf.h $S/coords/imp/main.h $S/coords/imp/type.h $S/coords/type.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/release/alloc.h $S/d/cuda/debug/alloc.h $S/d/cuda/imp.h $S/utils/msg.h
$B/io/mesh_read/edg/imp.o: $S/utils/imp.h $S/utils/error.h $S/io/mesh_read/edg/imp/main.h $S/algo/edg/imp.h $S/io/mesh_read/edg/imp/type.h $S/utils/msg.h
$B/io/mesh_read/imp.o: $S/utils/imp.h $S/utils/error.h $S/io/mesh_read/imp/ply.h $S/io/mesh_read/imp.h $S/io/mesh_read/imp/main.h $S/io/mesh_read/imp/type.h $S/io/mesh_read/edg/imp.h $S/io/mesh_read/imp/off.h $S/utils/msg.h
$B/io/vtk/imp.o: $S/io/vtk/imp/file.h $S/utils/os.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/io/vtk/imp/util.h $S/mpi/wrapper.h $S/io/vtk/imp.h $S/io/write/imp.h $S/algo/scalars/imp.h $S/io/vtk/imp/print.h $S/algo/key_list/imp.h $S/algo/vectors/imp.h $B/conf.h $S/io/vtk/imp/main.h $S/io/vtk/mesh/imp.h $S/io/vtk/imp/type.h $S/utils/msg.h
$B/io/vtk/mesh/imp.o: $S/utils/imp.h $S/utils/error.h $S/io/vtk/mesh/imp.h $S/io/mesh_read/imp.h $S/io/vtk/mesh/imp/main.h $S/io/vtk/mesh/imp/type.h $S/utils/msg.h
$B/io/write/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/io/write/imp.h $S/utils/mc.h $B/conf.h $S/io/write/imp/main.h $S/io/write/imp/type.h
$B/math/tform/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/math/tform/imp.h $B/conf.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/msg.h
$B/math/tri/imp.o: $S/math/tri/imp.h $S/math/tri/dev.h
$B/mesh/angle/imp.o: $S/utils/imp.h $S/utils/error.h $S/math/tri/imp.h $S/mesh/angle/imp.h $S/io/mesh_read/imp.h $S/algo/vectors/imp.h $S/mesh/angle/imp/main.h $S/mesh/angle/imp/type.h $S/utils/msg.h
$B/mesh/cylindrical/imp.o: $S/utils/imp.h $S/utils/error.h $S/mesh/cylindrical/imp/util.h $S/mesh/cylindrical/imp.h $S/algo/kahan_sum/imp.h $S/algo/vectors/imp.h $S/mesh/cylindrical/imp/main.h $S/mesh/cylindrical/imp/type.h $S/utils/msg.h
$B/mesh/edg_len/imp.o: $S/utils/imp.h $S/utils/error.h $S/mesh/edg_len/imp.h $S/io/mesh_read/imp.h $S/algo/vectors/imp.h $S/mesh/edg_len/imp/main.h $S/mesh/edg_len/imp/type.h $S/utils/msg.h
$B/mesh/eng_kantor/imp.o: $S/utils/imp.h $S/utils/error.h $S/math/tri/imp.h $S/mesh/eng_kantor/imp.h $S/algo/scalars/imp.h $S/io/mesh_read/imp.h $S/algo/vectors/imp.h $S/mesh/eng_kantor/imp/main.h $S/mesh/angle/imp.h $S/mesh/eng_kantor/imp/type.h $S/utils/msg.h
$B/mesh/scatter/imp.o: $S/utils/imp.h $S/utils/error.h $S/mesh/scatter/imp.h $S/algo/scalars/imp.h $S/io/mesh_read/imp.h $S/mesh/scatter/imp/main.h $S/mesh/scatter/imp/type.h $S/utils/msg.h
$B/mesh/vert_area/imp.o: $S/utils/imp.h $S/utils/error.h $S/math/tri/imp.h $S/mesh/vert_area/imp.h $S/io/mesh_read/imp.h $S/algo/vectors/imp.h $S/mesh/vert_area/imp/main.h $S/mesh/vert_area/imp/type.h $S/utils/msg.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/u/mesh/eng_kantor/main.o: $S/utils/imp.h $S/utils/error.h $S/conf/imp.h $S/mpi/wrapper.h $S/algo/scalars/imp.h $S/mesh/cylindrical/imp.h $S/io/mesh_read/imp.h $S/mesh/scatter/imp.h $S/algo/vectors/imp.h $S/mesh/eng_kantor/imp.h $S/utils/mc.h $S/io/vtk/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/os.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/nvtx/imp.o: $S/utils/error.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
$B/utils/string/imp.o: $S/utils/error.h $S/utils/string/imp.h

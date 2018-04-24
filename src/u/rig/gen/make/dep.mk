$B/algo/edg/imp.o: $S/utils/error.h $S/algo/edg/imp.h $S/algo/edg/imp/main.h
$B/algo/force_stat/imp.o: $S/utils/error.h $S/inc/type.h $S/algo/force_stat/imp.h $S/d/api.h $S/algo/force_stat/imp/main.h $S/utils/msg.h
$B/algo/kahan_sum/imp.o: $S/utils/imp.h $S/utils/error.h $S/algo/kahan_sum/imp.h $S/algo/kahan_sum/imp/main.h $S/algo/kahan_sum/imp/type.h
$B/algo/key_list/imp.o: $S/utils/imp.h $S/utils/error.h $S/algo/key_list/imp/util.h $S/algo/key_list/imp.h $S/utils/string/imp.h $S/algo/key_list/imp/main.h $S/algo/key_list/imp/type.h $S/utils/msg.h
$B/algo/minmax/imp.o: $S/inc/conf.h $S/inc/type.h $S/d/q.h $S/algo/minmax/imp.h $S/utils/cc.h $S/inc/def.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/utils/kl.h $S/d/ker.h $S/utils/msg.h
$B/algo/scalars/imp.o: $S/utils/imp.h $S/utils/error.h $S/algo/scalars/imp.h $S/algo/vectors/imp.h $S/algo/scalars/imp/main.h $S/algo/scalars/imp/type.h
$B/algo/scan/imp.o: $S/algo/scan/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/q.h $S/algo/scan/imp.h $S/utils/cc.h $S/algo/scan/dev.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/utils/kl.h $S/algo/scan/cuda/type.h $S/algo/scan/cpu/type.h $S/algo/scan/cuda/imp.h $S/d/ker.h
$B/algo/vectors/imp.o: $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/algo/vectors/imp.h $S/algo/vectors/imp/main.h $S/algo/vectors/imp/type.h $S/math/tform/imp.h $S/coords/imp.h
$B/clist/imp.o: $S/clist/imp/fin.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/struct/partlist/dev.h $S/clist/imp.h $S/algo/scan/imp.h $S/utils/cc.h $S/inc/def.h $S/clist/dev.h $S/clist/imp/ini.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/clist/imp/main.h $S/struct/partlist/type.h $S/clist/imp/type.h $S/utils/kl.h $S/clist/dev/main.h $S/utils/msg.h
$B/conf/imp.o: $S/utils/imp.h $S/utils/error.h $S/conf/imp/set.h $S/conf/imp.h $S/conf/imp/main.h $S/conf/imp/type.h $S/conf/imp/get.h $S/utils/msg.h
$B/coords/conf.o: $S/utils/imp.h $S/coords/ini.h $S/utils/error.h $S/conf/imp.h
$B/coords/imp.o: $S/utils/imp.h $S/inc/conf.h $S/coords/ini.h $S/utils/error.h $S/mpi/wrapper.h $S/coords/imp.h $S/utils/mc.h $B/conf.h $S/coords/imp/main.h $S/coords/imp/type.h $S/coords/type.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/imp.h
$B/flu/imp.o: $S/flu/imp/fin.h $S/utils/imp.h $S/utils/os.h $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/mpi/wrapper.h $S/flu/imp.h $S/algo/scan/imp.h $S/flu/imp/cells.h $S/utils/cc.h $S/inc/def.h $S/io/restart/imp.h $S/flu/imp/generate.h $S/flu/imp/ini.h $S/io/txt/imp.h $S/inter/color/imp.h $S/utils/mc.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/struct/partlist/type.h $S/clist/imp.h $S/flu/imp/start.h $S/flu/imp/txt.h $S/coords/imp.h $S/utils/msg.h
$B/inter/color/conf.o: $S/utils/imp.h $S/utils/error.h $S/conf/imp.h $S/inter/color/imp.h
$B/inter/color/imp.o: $S/utils/imp.h $S/inc/conf.h $S/inter/color/imp/drop.h $S/utils/error.h $S/inc/type.h $S/inter/color/imp.h $S/utils/cc.h $S/inc/def.h $S/inter/color/imp/unif.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/inter/color/imp/main.h $S/inter/color/imp/type.h $S/coords/imp.h $S/utils/msg.h
$B/io/restart/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/mpi/wrapper.h $S/io/restart/imp.h $S/inc/def.h $B/conf.h $S/io/restart/imp/main.h $S/coords/imp.h $S/utils/msg.h
$B/io/txt/imp.o: $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/io/txt/imp.h $S/io/txt/imp/dump.h $S/io/txt/imp/read.h $S/io/txt/imp/type.h $S/utils/msg.h
$B/math/linal/imp.o: $S/utils/error.h $S/math/linal/imp.h
$B/math/rnd/imp.o: $S/utils/imp.h $S/utils/error.h $S/math/rnd/imp.h
$B/math/tform/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/math/tform/imp.h $B/conf.h $S/math/tform/imp/main.h $S/math/tform/imp/type.h $S/math/tform/type.h $S/utils/msg.h
$B/math/tri/imp.o: $S/math/tri/imp.h $S/math/tri/dev.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/struct/farray/imp.o: $S/struct/farray/imp.h $S/struct/farray/imp/main.h $S/struct/farray/type.h
$B/struct/parray/imp.o: $S/struct/parray/imp.h $S/struct/parray/imp/main.h $S/struct/parray/type.h
$B/struct/pfarrays/imp.o: $S/struct/farray/imp.h $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/struct/pfarrays/imp.h $S/struct/parray/imp.h $S/struct/pfarrays/imp/main.h $S/struct/pfarrays/imp/type.h
$B/u/rig/gen/main.o: $S/flu/imp.h $S/inc/type.h $S/mpi/wrapper.h $S/utils/mc.h $S/clist/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/error.h $S/utils/imp.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/nvtx/imp.o: $S/utils/error.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
$B/utils/string/imp.o: $S/utils/error.h $S/utils/string/imp.h

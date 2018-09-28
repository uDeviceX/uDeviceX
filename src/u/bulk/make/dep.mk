$B/algo/convert/imp.o: $S/inc/conf.h $S/inc/type.h $S/algo/convert/imp.h $S/inc/dev.h $B/conf.h $S/algo/convert/imp/main.h $S/utils/kl.h $S/algo/convert/dev/main.h
$B/algo/scan/imp.o: $S/algo/scan/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/q.h $S/algo/scan/imp.h $S/utils/cc.h $S/algo/scan/dev.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/utils/kl.h $S/algo/scan/cuda/type.h $S/algo/scan/cpu/type.h $S/algo/scan/cuda/imp.h $S/d/ker.h
$B/clist/imp.o: $S/clist/imp/fin.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/struct/partlist/dev.h $S/clist/imp.h $S/algo/scan/imp.h $S/utils/cc.h $S/inc/def.h $S/clist/dev.h $S/clist/imp/ini.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/clist/imp/main.h $S/struct/partlist/type.h $S/clist/imp/type.h $S/utils/kl.h $S/struct/particle/dev.h $S/clist/dev/main.h $S/utils/msg.h
$B/conf/imp.o: $S/utils/imp.h $S/utils/error.h $S/conf/imp/set.h $S/conf/imp.h $S/conf/imp/main.h $S/conf/imp/type.h $S/conf/imp/get.h $S/utils/msg.h
$B/coords/conf.o: $S/utils/imp.h $S/coords/ini.h $S/utils/error.h $S/conf/imp.h
$B/coords/imp.o: $S/utils/imp.h $S/inc/conf.h $S/coords/ini.h $S/utils/error.h $S/mpi/wrapper.h $S/coords/imp.h $S/utils/mc.h $B/conf.h $S/coords/imp/main.h $S/coords/imp/type.h $S/coords/type.h
$B/d/api.o: $S/d/cpu/imp.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/d/common.h $S/d/api.h $B/conf.h $S/d/cuda/release/alloc.h $S/d/cuda/debug/alloc.h $S/d/cuda/imp.h $S/utils/msg.h
$B/fluforces/bulk/imp.o: $S/struct/farray/imp.h $S/inc/conf.h $S/struct/farray/type.h $S/utils/error.h $S/inc/type.h $S/fluforces/bulk/dev/fetch.h $S/fluforces/bulk/imp.h $S/math/rnd/dev.h $S/utils/cc.h $S/inc/def.h $S/math/rnd/imp.h $S/pair/imp.h $S/pair/type.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/fluforces/bulk/imp/main.h $S/utils/texo.h $S/fluforces/bulk/imp/type.h $S/pair/dev.h $S/utils/texo.dev.h $S/utils/kl.h $S/math/dev.h $S/struct/farray/dev.h $S/fluforces/bulk/dev/main.h $S/d/ker.h
$B/fluforces/halo/imp.o: $S/fluforces/halo/dev/map.h $S/flu/type.h $S/struct/farray/imp.h $S/inc/conf.h $S/struct/farray/type.h $S/utils/error.h $S/struct/parray/dev.h $S/inc/type.h $S/d/q.h $S/fluforces/halo/imp.h $S/struct/parray/imp.h $S/math/rnd/dev.h $S/utils/cc.h $S/inc/def.h $S/pair/imp.h $S/pair/type.h $S/inc/dev.h $S/d/api.h $B/conf.h $S/fluforces/halo/imp/main.h $S/fluforces/halo/imp/type.h $S/utils/kl.h $S/pair/dev.h $S/frag/dev.h $S/math/dev.h $S/struct/farray/dev.h $S/struct/parray/type.h $S/frag/imp.h $S/fluforces/halo/dev/main.h $S/fluforces/halo/dev/dbg.h $S/d/ker.h
$B/fluforces/imp.o: $S/fluforces/imp/fin.h $S/flu/type.h $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/inc/type.h $S/mpi/wrapper.h $S/fluforces/imp.h $S/fluforces/halo/imp.h $S/fluforces/bulk/imp.h $S/algo/convert/imp.h $S/struct/parray/imp.h $S/utils/cc.h $S/math/rnd/imp.h $S/fluforces/imp/ini.h $S/utils/mc.h $S/d/api.h $S/inc/dev.h $B/conf.h $S/fluforces/imp/main.h $S/fluforces/imp/type.h $S/utils/kl.h $S/frag/imp.h $S/utils/msg.h
$B/frag/imp.o: $S/frag/dev.h $S/frag/imp.h
$B/io/txt/imp.o: $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/io/txt/imp.h $S/io/txt/imp/dump.h $S/io/txt/imp/read.h $S/io/txt/imp/type.h $S/utils/msg.h
$B/math/rnd/imp.o: $S/utils/imp.h $S/utils/error.h $S/math/rnd/imp.h
$B/mpi/glb.o: $S/inc/conf.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/mpi/wrapper.o: $S/mpi/wrapper.h
$B/pair/conf.o: $S/utils/error.h $S/conf/imp.h $S/pair/imp.h $S/inc/def.h $S/pair/type.h
$B/pair/imp.o: $S/utils/imp.h $S/inc/conf.h $S/utils/error.h $S/pair/imp.h $S/inc/def.h $S/pair/imp/main.h $S/pair/imp/type.h $S/pair/type.h
$B/struct/farray/imp.o: $S/struct/farray/imp.h $S/struct/farray/imp/main.h $S/struct/farray/type.h
$B/struct/parray/imp.o: $S/struct/parray/imp.h $S/struct/parray/imp/main.h $S/struct/parray/type.h
$B/struct/pfarrays/imp.o: $S/struct/farray/imp.h $S/utils/imp.h $S/utils/error.h $S/inc/type.h $S/struct/pfarrays/imp.h $S/struct/parray/imp.h $S/struct/pfarrays/imp/main.h $S/struct/pfarrays/imp/type.h
$B/u/bulk/main.o: $S/flu/type.h $S/struct/farray/imp.h $S/utils/imp.h $S/inc/conf.h $S/fluforces/imp.h $S/utils/error.h $S/inc/type.h $S/conf/imp.h $S/mpi/wrapper.h $S/struct/parray/imp.h $S/coords/ini.h $S/utils/cc.h $S/io/txt/imp.h $S/pair/imp.h $S/inc/dev.h $S/utils/mc.h $S/d/api.h $B/conf.h $S/struct/partlist/type.h $S/clist/imp.h $S/coords/imp.h $S/mpi/glb.h $S/utils/msg.h
$B/utils/cc.o: $S/utils/cc/common.h $S/inc/conf.h $S/utils/error.h $S/d/api.h $B/conf.h
$B/utils/error.o: $S/utils/msg.h $S/utils/error.h
$B/utils/imp.o: $S/utils/os.h $S/utils/error.h $S/utils/imp.h $S/utils/msg.h
$B/utils/mc.o: $S/inc/conf.h $S/utils/error.h $S/mpi/wrapper.h $S/utils/mc.h $B/conf.h
$B/utils/msg.o: $S/utils/msg.h
$B/utils/nvtx/imp.o: $S/utils/error.h
$B/utils/os.o: $S/utils/os.h $S/utils/error.h $S/utils/msg.h
$B/utils/string/imp.o: $S/utils/error.h $S/utils/string/imp.h

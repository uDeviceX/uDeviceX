$B/conf/imp.o: $S/conf/imp.cpp; $X -I$S/. -I$S/conf
$B/coords/conf.o: $S/coords/conf.cpp; $X -I$S/. -I$S/coords
$B/coords/imp.o: $S/coords/imp.cpp; $X -I$S/. -I$S/coords
$B/d/api.o: $S/d/api.cu; $N -I$S/. -I$S/d
$B/dbg/conf.o: $S/dbg/conf.cpp; $X -I$S/. -I$S/dbg
$B/dbg/imp.o: $S/dbg/imp.cu; $N -I$S/. -I$S/dbg
$B/io/txt/imp.o: $S/io/txt/imp.cpp; $X -I$S/. -I$S/io/txt
$B/mpi/glb.o: $S/mpi/glb.cpp; $X -I$S/. -I$S/mpi
$B/mpi/type.o: $S/mpi/type.cpp; $X -I$S/. -I$S/mpi
$B/mpi/wrapper.o: $S/mpi/wrapper.cpp; $X -I$S/. -I$S/mpi
$B/u/dbg/main.o: $S/u/dbg/main.cu; $N -I$S/.
$B/utils/cc.o: $S/utils/cc.cpp; $X -I$S/. -I$S/utils
$B/utils/error.o: $S/utils/error.cpp; $X -I$S/. -I$S/utils
$B/utils/imp.o: $S/utils/imp.cpp; $X -I$S/. -I$S/utils
$B/utils/mc.o: $S/utils/mc.cpp; $X -I$S/. -I$S/utils
$B/utils/msg.o: $S/utils/msg.cpp; $X -I$S/. -I$S/utils
$B/utils/os.o: $S/utils/os.cpp; $X -I$S/. -I$S/utils

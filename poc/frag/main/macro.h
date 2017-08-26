#define  frag_to_id(x, y, z) (frag_id000[ (x)+1][ (y)+1][ (z)+1])
#define frag_fro_id(x, y, z) (frag_id000[-(x)+1][-(y)+1][-(z)+1])
#define frag_ncell0(x, y, z) \
    ((((x) == 0 ? (XS) : 1)) * \
     (((y) == 0 ? (YS) : 1)) * \
     (((z) == 0 ? (ZS) : 1)))
#define frag_ncell(i) \
    (frag_ncell0(frag_to_dir[(i)][0], frag_to_dir[(i)][1], frag_to_dir[(i)][2]))
#define frag_bulk (27)

# hiwi for solvent

# problems
* `odstr` uses hardcoded tags for MPI
* `pp0` is swaped with `pp`
* two layer with own states `x` and `odstr`

# from x/x.decl.h

    MPI_Comm cart;
    uchar4 *subi_lo, *subi_re;
    uint   *iidx;
    bool first = true;
    MPI_Request send_size_req[27], recv_size_req[27],
            send_mesg_req[27], recv_mesg_req[27];
    int rank[27];

    Particle *pp_re;
    unsigned char *count_zip;
    Particle *pp0;

# from odstr.decl

    namespace s {
      int **iidx;
      float2 **dev;
      float *hst[27];

      int *size_dev, *strt;
      int size[27];
      PinnedHostBuffer4<int>* size_pin;

      float2 *hst_[27];
      int    *iidx_[27];
     }

     namespace r {
      float2 **dev;
      int *strt, *strt_pa;
      int tags[27];
      int    size[27];
      float *hst[27];
      float2 *hst_[27];
     }

# T
zip0, zip1, cells

# Q
pp, n

namespace sdstr {
  namespace s { /* [s]end */
    int **iidx; /* indices */
    float2 **dev;   /* buffers */
    float *hst[27];

    int *size_dev, *strt;
    int size[27];
    PinnedHostBuffer4<int>* size_pin;

    float2 *hst_[27];
    int    *iidx_[27];
  }

  namespace r { /* [r]ecive */
    float2 **dev;
    int *strt, *strt_pa; /* _pa: padded */
    int tags[27];
    int    size[27];
    float *hst[27];
    float2 *hst_[27];
  }
}

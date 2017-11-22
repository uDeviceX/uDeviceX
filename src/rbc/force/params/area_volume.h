#ifndef RBCtotArea
  #if      RBCnv==498
    #define RBCtotArea (2.240e-4*(RBCnv)*(RBCnv))
  #else
    #define RBCtotArea (2.3835e-04*(RBCnv)*(RBCnv))
  #endif
#endif

#ifndef RBCtotVolume
  #if      RBCnv==498
    #define RBCtotVolume (2.185e-7*(RBCnv)*(RBCnv)*(RBCnv))
  #else
    #define RBCtotVolume (2.2670e-07*(RBCnv)*(RBCnv)*(RBCnv))
  #endif
#endif

/* TODO: do not use a hack for 498 */

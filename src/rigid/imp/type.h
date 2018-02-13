struct RigPinInfo {
    int3 com;  /* what components of com stay fixed */
    int3 axis;  /* what components of rotation stay fixed */
    int pdir; /* direction of periodicity */
};

struct DCont {
    int ncells;  // cells to be controlled
    int *cids;   // cell ids of the controlled cells

    curandState_t *rnd; // random states
};

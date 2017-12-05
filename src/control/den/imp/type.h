struct DCont {
    int *ndead_dev, ndead;
    int *kk;
    // curandState_t *rnd; // random states
};

struct DContMap {
    int n;     // cells to be controlled
    int *cids; // cell ids of the controlled cells
};

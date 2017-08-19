namespace rex {
struct LocalHalo {
    int n;
    int* indexes;
    PinnedHostBuffer<Force>* ff;
};

}

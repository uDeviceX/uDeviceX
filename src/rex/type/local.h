namespace rex {
struct LocalHalo {
    int* indexes;
    PinnedHostBuffer<Force>* ff;
};

}

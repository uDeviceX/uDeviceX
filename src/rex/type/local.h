namespace rex {
struct LocalHalo {
    int n;
    int* indexes;
    PinnedHostBuffer<Force>* ff;

    Force* ff0;
    Force* ff_pi; /* pinned */
};

}

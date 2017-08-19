namespace rex {
struct LocalHalo {
    int n;
    int* indexes;

    Force* ff;
    Force* ff_pi; /* pinned */
};

}

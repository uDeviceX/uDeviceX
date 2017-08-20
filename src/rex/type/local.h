namespace rex {
struct LocalHalo {
    int* indexes;

    Force* ff;
    Force* ff_pi; /* pinned */
};

}

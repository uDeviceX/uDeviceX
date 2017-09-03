namespace rex {
struct LFrag {
    int* indexes;

    Force* ff;
    Force* ff_pi; /* pinned */
};

struct RFrag {
    Force* ff_pi; /* pinned */
};
}

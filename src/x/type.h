namespace x {
struct TicketCom { /* tikcket communiction */
    MPI_Comm cart;
    int ranks[26];
};

struct TicketR { /* ticket receive */
    int tags[26];
};

struct TicketTags { /* basetags */
    int btc, btp1, btp2, btf;
};

struct TicketPack { /* helps pack particles for mpi */
    int *counts, *starts, *offsets;
    int *tstarts; /* total start */
    PinnedHostBuffer5<int> *tstarts_hst;
    PinnedHostBuffer5<int> *offsets_hst;
};

}

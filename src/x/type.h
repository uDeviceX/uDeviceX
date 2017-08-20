namespace x {
struct TicketCom { /* ticket communiction */
    MPI_Comm cart;
    int ranks[26];
};

struct TicketR { /* ticket receive */
    int tags[26];
};

struct TicketTags { /* basetags */
    int btc, btp1, btp2, btf;
};

struct TicketPack { /* helps pack particles (device) */
    int *counts, *starts, *offsets;
    int *tstarts; /* total start */
};

struct TicketPinned { /* helps pack particles (host) */
    int *tstarts;
    int *counts;
};

}

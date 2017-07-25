namespace x {
struct TicketCom { /* tikcket communiction */
    MPI_Comm cart;
    int ranks[26];
};

struct TicketR { /* ticket receive */
    int tags[26];
};

struct TicketTags {
    /* basetags */
    int btc, btp1, btp2, btf;
};

struct TicketPack {
    DeviceBuffer<int> *packscount, *packsstart, *packsoffset, *packstotalstart;
    PinnedHostBuffer1<int> *host_packstotalstart, *host_packstotalcount;
};

}

namespace x {
struct TicketCom { /* tikcket communiction */
    MPI_Comm cart;
    int ranks[26];
};

struct TikcketR { /* ticket receive */
    int tags[26];
};

}

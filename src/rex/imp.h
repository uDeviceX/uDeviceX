namespace rex {
void ini(LFrag *local);
void fin();
void clear(int nw, x::TicketPack tp);
void scanA(ParticlesWrap *w, int nw, x::TicketPack tp);
void scanB(int nw, x::TicketPack tp);
void copy_offset(int nw, x::TicketPack tp, x::TicketPinned ti);
void copy_starts(x::TicketPack tp, /**/ x::TicketPinned ti);
void pack(ParticlesWrap *w, int nw, x::TicketPack tp, Particle *buf);
void recvF(int ranks[26], int tags[26], x::TicketTags t, int counts[26], LFrag *local);
void recvC(int ranks[26], int tags[26], x::TicketTags t, int counts[26]);
void recvP(int ranks[26], int tags[26], x::TicketTags t, int counts[26], Pap26 PP_pi);
void sendC(int dranks[26], x::TicketTags t, int counts[26]);
void sendP(int ranks[26], x::TicketTags t, x::TicketPinned ti, Particle *pp, int counts[26]);
void sendF(int ranks[26], x::TicketTags t, int counts[26], Fop26 FF_pi);
void unpack(ParticlesWrap *w, int nw, x::TicketPack tp);

namespace s { /* send */
void waitC();
void waitP();
void waitA();
}

namespace r { /* recive */
void waitC();
void waitP();
void waitA();
}

} /* namespace */

namespace x {
Pap26 PP;
Fop26 FF;
Pap26 PP_pi;

rex::RFrag remote[26];
rex::LFrag local[26];

TicketCom    tc;
TicketR      tr;
TicketTags   tt;
TicketPack   tp;
TicketPinned ti;

int first;
Particle *buf;
Particle *buf_pi;

int recv_counts[26];
}

# hiwi for solvent

# problems
* in `odstr` tags for MPI are hardcoded
* `pp0` is swaped with `pp`
* two layers with states: `x` and `odstr`

# Q

	Particle *pp
	int       n
	Clist *cells

# T1

	float4  *zip0;
	ushort4 *zip1;

# T2

	MPI_Comm cart
	MPI_Request send_size_req[27], recv_size_req[27]
	MPI_Request send_mesg_req[27], recv_mesg_req[27]
	int rank[27]
	bool first = true
	Odstr odstr

# W

	uchar4 *subi_lo, *subi_re
	uint   *iidx
	Particle *pp_re
	unsigned char *count_zip
	Particle *pp0

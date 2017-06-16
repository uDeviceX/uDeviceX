# hiwi for solvent

# problems
* in `odstr` tags for MPI are hardcoded
* `pp0` is swaped with `pp`
* two layers with states: `x` and `odstr`

# T
`zip0`, `zip1`, `cells`

	MPI_Comm cart
	int rank[27]
	bool first = true
	MPI_Request send_size_req[27], recv_size_req[27]

# W

	uchar4 *subi_lo, *subi_re
	uint   *iidx
	MPI_Request send_mesg_req[27], recv_mesg_req[27]

	Particle *pp_re
	unsigned char *count_zip
	Particle *pp0

# Q
`pp`, `n`

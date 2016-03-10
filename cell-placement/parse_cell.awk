#!/usr/bin/awk -f

# Parse cell template
# Usage:
# ./cell-placement/parse_cell.awk cuda-rbc/rbc.dat > cuda-rbc/rbc2.atom_parsed

{
    gsub(/^[A-Za-z]+/, "") # get rid of words (Angles, Bonds,
			   # Dihedrals ..)
}

/^[ ^t]*$/ {          # collapse several empty lines into one
    if (!prev_empty)
	print
    prev_empty = 1
    next
}

{
    prev_empty = 0
    $1=$1             # reformat the string
    print
}

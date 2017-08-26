# example

body () { echo ; ex/0 $x $y $z; }

for x in -1 0 1
do for y in -1 0 1
   do for z in -1 0 1
      do body
      done
   done
done

# TEST: ex.t0
# { make clean; make; } > /dev/null
# ex/0.sh               > ex.out.txt
#

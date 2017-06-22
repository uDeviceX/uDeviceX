make > /dev/null

#u.conf test_data/double.poiseuille.h <<!
#pushflow # hmmm
# now run
#run
#!

#u.conf test_data/double.poiseuille.h <<!
#run
#pushflow # hmmm
#run
#!

u.conf ../src test_data/double.poiseuille.h <<!
a=1
run
b=2
pushflow # hmmm
run
!


#u.conf test_data/double.poiseuille.h <<!
#pushflow # hmmm
#!

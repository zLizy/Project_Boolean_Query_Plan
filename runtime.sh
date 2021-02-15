#!/bin/sh
# chmod +x runtime.sh
# ./runtime.sh
for i in {1,2,3,4}
do 
	M=$((2**$i))
	M=$((100*$M))
	for N in {20,40,80}
	do
		for q in {50,100,150,200}
		do
			python run.py -m $M -n $N -nquery $q -constraint cost -bound 200 
		done
	done
done


for i in 0 1 2 3 4;
do
	for mdist in 'uniform' 'power_law';
	do
		for qdist in 'uniform' 'power_law';
		do
			for bound in 100 150 50;
			do
				echo ${mdist} ${qdist}
				python run.py -mdist ${mdist} -qdist ${qdist} -constraint cost -bound ${bound} -synthetic -record -approach baseline
				python run.py -mdist ${mdist} -qdist ${qdist} -constraint cost -bound ${bound} -synthetic -record
				python run.py -mdist ${mdist} -qdist ${qdist} -constraint cost -bound ${bound} -synthetic -order -record
			done
		done
	done
done


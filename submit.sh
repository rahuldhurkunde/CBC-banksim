if [ $# -eq 0 ]
then
	echo "Provide nworkflow"
	exit 1
else
	let end=$1-1
	echo "nworkflow $1 and last ind of loop $end"
	
	for i in $(seq 0 $end); do
		submit_dir="part_$i/"
		echo "Submitting in sub dir $submit_dir"
		cd $submit_dir

		pycbc_submit_dax --no-create-proxy --no-grid \
		--local-dir ./ \
		--no-query-db   
		
		cd ../
	done
fi

#!/bin/bash

dir='/work/rahul.dhurkunde/searches/eccentric-bns-search/banksim'

first=0
last=10

if [ $# -eq 0 ]
then
    echo "Provide sub-directory name"
    exit 1
else
    echo "Executing in directory $1"
    echo "Injs from $first to $last"
    cp submit.sh $1
	cp combine_FFs.py $1
	cd $1
    mkdir results

    $dir/FF_parser --config-files $dir/workflow.ini \
                            --template_bank /work/rahul.dhurkunde/searches/eccentric-bns-search/banks/spinecc/mtotal-10_ecc-0.28_spin-0.1/small_bank_sorted.hdf \
                            --inj_file $dir/test.hdf \
                            --psd_file $dir/o3psd.txt \
                            --first $first \
                            --last $last 
    #./submit.sh
fi


#!/bin/bash

dir='/work/rahul.dhurkunde/searches/banksim'

first=0
last=10

workflow_config=$dir/example_workflow.ini

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

    $dir/FF_parser --config-files $workflow_config \
                            --first $first \
                            --last $last 
    #./submit.sh
fi


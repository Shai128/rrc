#!/bin/bash

files=(2011_09_26_drive_0051
2011_09_26_drive_0057
2011_09_26_drive_0059
2011_09_26_drive_0096
2011_09_26_drive_0104
2011_09_29_drive_0071
2011_09_30_drive_0018
2011_09_30_drive_0028
)

for i in ${files[@]}; do
        if [ ${i:(-3)} != "zip" ]
        then
                shortname=$i'_sync.zip'
                fullname=$i'/'$i'_sync.zip'
        else
                shortname=$i
                fullname=$i
        fi
	echo "Downloading: "$shortname
        wget 'https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/'$fullname
        unzip -o $shortname
        rm $shortname
done

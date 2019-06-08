#!/bin/sh
shopt -s globstar
if [ ! -e /unzipped ];
then 
	mkdir /unzipped
fi
for i in $1/**/*.*z; 
do
	if [ ! -e /unzipped/$(basename $(dirname $i)) ]; 
	then 
		mkdir /unzipped/$(basename $(dirname $i))
	fi
#do echo ${i#*/}
#do echo $(basename $(dirname $i))
 	gunzip -c "$i" > "/unzipped/$(basename $(dirname $i))/${i##*/}.txt" 
done

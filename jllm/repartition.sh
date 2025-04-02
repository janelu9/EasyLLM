#! /bin/bash
# @author Jian Lu 
# @date 2023/8/9
set -e
num=$2
if [ "$num" == "" ];then
	num=`du -shB1G $1|cut -f1`
fi
cd $1
cur_partitions=`ls`
uuid=`head -1 /dev/urandom|md5sum|head -c18`
for ((i=0;i<$num;i++));do
	file=`printf "$uuid-part-%05d" $i`
	rm -rf $file
	mkdir $file
done
i=0
for f in `find -name *.parquet|sort`;do
	mv $f `printf "$uuid-part-%05d/" $((i%$num))`
	i=$((i+1))
done
rm -rf $cur_partitions 
crc=`ls .*.crc`
cat $crc | awk '{
    sum+=$1; 
    if($2 > max || NR==1) max=$2; 
    last3=$3; 
    last4=$4
} 
END {
    print sum" "max" "last3" "last4
}' >.$uuid.crc
rm -f $crc
echo -e "Data: $1\nPartitions: $num\nFiles: $i\nSamples: $(cat .$uuid.crc|awk '{print $1"*"$2}')\nUUID: $uuid\n\nSize\tPartition">data.info
du -sh *-part-* >>data.info

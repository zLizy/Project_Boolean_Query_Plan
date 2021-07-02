oldifs="$IFS"
IFS=$'\n'

pod=`argo list | awk '{print $1}' | sed -n '2,$p'`

array=($pod)

for element in "${array[@]}"
do
    argo delete ${element}
done

IFS="$oldifs"




oldifs="$IFS"
IFS=$'\n'

pod=`argo list | awk '{print $1}' | sed -n '2,$p'`

duration=`argo list | awk '{print $4}' | sed -n '2,$p'`

# IFS=' ' read -r -a array <<< "$pod"
array=($pod)

echo $1
[ -d "log/$1" ] &&  echo "Directory log/$1 found." || mkdir log/$1

for element in "${array[@]}"
do
    argo get ${element} -o json > log/$1/${element}.json
    argo delete ${element}
done

IFS="$oldifs"



workdir=$(pwd)

query="(object=car and color=red) or (object=truck and color=yellow);"
# query="object=car;"
echo $query
constrint="accuracy"
echo $constrint

# 分隔符
IFS_OLD=$IFS
IFS=$'｜' 

# get optimal models
cd query2optimization
mvn exec:java -Dexec.mainClass="Main" -Dquery=$query -Dconstraint=$constraint

IFS=$IFS_OLD

# get optimal execution order
cd $workdir
python run.py --query $query


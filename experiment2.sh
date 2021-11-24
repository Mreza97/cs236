
exec &> >(tee -a experiment2.out)

echo "

*** STARTING EXPERIMENT 2 ***

date: $(date)
args: "${@}"


"

python3 experiment2.py "${@}"
#sudo /sbin/shutdown -h now


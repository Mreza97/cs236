
exec &> >(tee -a experiment1.out)

echo "\n\n\n *** STARTING EXPERIMENT 1 ***\n$(date)\n\n\n"

python3 experiment1.py
sudo /sbin/shutdown -h now


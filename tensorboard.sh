
( ls -tr /tmp/.tensorboard-info/pid-*.info | tail -1 | sed 's/^.*pid-\([0-9]*\).info$/\1/' | xargs -n1 kill ;) &>/dev/null

sudo nohup /opt/conda/bin/tensorboard --logdir logs --port 443 --host 0.0.0.0 >> tensorboard.log 2>&1 &

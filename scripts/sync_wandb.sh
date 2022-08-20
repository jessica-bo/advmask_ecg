#!/bin/sh
export LC_ALL=C.UTF-8
export LANG=C.UTF-8

for d in /home/gridsan/ybo/advaug/outputs/transfer/linear/adversarial_chapman/* ; do
    echo "$d/wandb"
    python -m wandb sync --include-offline $d/seed0/wandb/offline-*

    # mv ${d} ${d/lr0.0001_bs256/lr0.0001_bs32}
    # mv "$d" "${d/_h.png/_half.png}"

    # rm -r $d/seed10
    # tail $d/run_log.sh.log*
    # mv $d/run* $d/seed1/
    # mv $d/sweep* $d/seed1/
done


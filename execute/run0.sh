for method in fedavg krum trimmed_mean fang
do
    python main.py --gpu 0 --method $method --tsboard --c_frac 0.0 --quantity_skew
done
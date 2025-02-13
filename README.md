# Performative Prediction on Games and Mechanism Design

Repository for "[Performative Prediction on Games and Mechanism Design](https://arxiv.org/abs/2408.05146)". AISTATS 2025. Góis, A., Mofakhami, M., Santos, F. P., Gidel, G., & Lacoste-Julien, S.

## Instructions

Simulation for RRM with scale-free networks (Figure 4):
- `python plot_rrm_scalefree.py --save_path rrm`

Heatmaps for the anarchic setting with \tau=0 (Figure 5):
- `python plot_alpha_heatmap.py --graph full`
- `python plot_alpha_heatmap.py --graph scale-free`

Trust oscillation (Figure 6):
- `python plot_trust_oscillation.py --discount_rate med`

Tradeoffs between accuracy and welfare (Figure 7):
- `python plot_tradeoff_thresholds.py`

Plot to compare architectures (Figure 8):
- `python train_all_archs.py --seed 0 --stats_path crd_archs_stats --loss group`
  - repeat for seeds 1, 2, 3
- `python plot_architectures_avgs.py --read_path crd_archs_stats --loss group`

Histograms for RRM with scale-free networks in appendix (Figure 12):
- `python plot_rrm_scalefree.py --plot_single_pop -n 20`
  - repeat for n=30, 50

Ablation of gradient components in appendix (Figure 15):
- `python train.py --architecture gnn+mlp --seed 0 --epochs 200 --topology scale-free --loss individual -lr 1e-4 --save_stats --use_custom_grad --block_prev_trust  --stats_path crd_stats_grad-blockprevtrust/`
  - repeat for seeds 1, 2, 3, 4
- `python train.py --architecture gnn+mlp --seed 0 --epochs 200 --topology scale-free --loss individual -lr 1e-4 --save_stats --use_custom_grad --block_trust_grad  --stats_path crd_stats_grad-blocktrust/`
  - repeat for seeds 1, 2, 3, 4
- `python train.py --architecture gnn+mlp --seed 0 --epochs 200 --topology scale-free --loss individual -lr 1e-4 --save_stats --use_custom_grad --stats_path crd_stats_grad-full/`
  - repeat for seeds 1, 2, 3, 4
- `python plot_ablation.py`

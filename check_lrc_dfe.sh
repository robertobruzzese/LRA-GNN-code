VAL="${1:-embeddings_ablation_morph_lrc_dfe/val}"
find "$VAL" -mindepth 1 -maxdepth 1 -type d | while read d; do
  lrc=$(ls "$d"/graph_lrc_*.pt 2>/dev/null | wc -l | tr -d ' ')
  dfe_g=$(ls "$d"/graph_rw.pt 2>/dev/null | wc -l | tr -d ' ')
  dfe_x=$(ls "$d"/deep_features_from_rw.pt "$d"/deep_features.pt 2>/dev/null | wc -l ' ' )
  [ "$lrc" = "8" ] && [ "$dfe_g" = "1" ] && [ "$dfe_x" = "1" ] || \
    echo "$(basename "$d"): LRC=$lrc/8 DFE_graph=$dfe_g/1 DFE_feats=$dfe_x/1"
done

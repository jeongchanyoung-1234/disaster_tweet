# disaster_tweet

- train_segmentedv2 -> shuf
 senmented와 cleaned로 다시 한번 accuracy 확인

- fasttext classifier 사용 시 78.6

```
python train.py --model_fn model.pth --file_path ./data/train_segmented.tsv --batch_size 64 --n_epochs 20 --embedding_dim 100 --num_layers 4 --t
rain_ratio 0.8 --dropout .4 --hidden_size 64
```

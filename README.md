# VAE Vs Others

To run the file you run 

python main.py.

If you want to you can adjust all of the params needed to experiment.

```
parser.add_argument('--input_dim', type=int, default=784, help='Input dimension')
parser.add_argument('--hidden_dim', type=int, default=200, help='Hidden dimension')
parser.add_argument('--z_dim', type=int, default=20, help='Latent dimension')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (Karpathy constant)')
parser.add_argument('--add_noise', action='store_true', help='Add noise to inputs')
parser.add_argument('--noise_factor', type=float, default=0.3, help='Noise factor')
parser.add_argument('--model_save_path', type=str, default='vae_model.pth', help='Path to save the model')
parser.add_argument('--inference_save_path', type=str, default='inference_reconstruction.png', help='Path to save the inference image')
parser.add_argument('--num_samples', type=int, default=8, help='Number of samples for inference')
parser.add_argument('--dataset_save_path', type=str, default='generated_dataset.pt', help='Path to save the generated dataset')
```

An example if I wanted to train for more epochs I would run.
```
python main.py --num_epochs 10
```
Note that `add_noise` will be true if you just include it in the command. 

# Trajectory Prediction with CVAE and MI-VAE

Two trained models, two testing scripts, one goal: roll out plausible state trajectories from a learned latent space.

---

## What's in this repo

| File | Description |
|---|---|
| `CVRNN_teacher_forced.py` | Trains the RNN-informed CVAE on real noise data |
| `MIVRNN_teacher_forced.py` | Trains the MI-informed CVAE with shared/private latent structure |
| `CVRNN_teacher_forced_test.py` | Loads `stepwise_cvae.pth` and rolls out trajectory samples |
| `MIVRNN_teacher_forced_test.py` | Loads `stepwise_mivae.pth` and rolls out trajectory samples |

---

## Models

### CVAE (`stepwise_cvae.pth`)

A conditional variational autoencoder trained to predict next states step by step. The encoder compresses full trajectories (state + control) into a latent code `z`, and the decoder unrolls predictions autoregressively using that code.

- State dim: 9, Control dim: 3
- Latent dim: 32, Hidden dim: 64
- KL annealing over first 300 epochs, max weight 0.05
- Trained on 200 real-noise trajectories

### MI-VAE (`stepwise_mivae.pth`)

A two-encoder CVAE trained on real (`A`) and simulated (`B`) noise trajectories simultaneously. The setup splits the latent space into two parts:

- `z1` — domain-specific (different prior per dataset)
- `z2` — shared across domains

After a warmup period, mutual information between `z1_A` and `z1_B` is maximized to push the two domains apart in that subspace. Reconstruction happens through separate decoders for each domain.

- z1 dim: 32, z2 dim: 32, Hidden dim: 324
- MI warmup: 50 epochs, beta (MI weight): 20.0
- 25 real + 500 simulated trajectories

---

## Running the test scripts

Both scripts load a checkpoint, pull one test trajectory from disk, and plot several sampled rollouts against ground truth.

**CVAE test:**
```bash
python test_cvrnn.py
```

**MI-VAE test:**
```bash
python test_mivae.py
```

Both scripts hardcode the data path and trajectory index. You'll want to update these two lines before running:

```python
# test_cvrnn.py
A_path = "path/to/your/data_process_real_noise"
df = pd.read_csv(csv_files[301], ...)  # change index to a test sample

# test_mivae.py
A_path = "path/to/your/data_process_real_noise"
full_data = load_test_traj(A_path, ..., idx=30)  # change idx
```

---

## Output

Each script generates a 2D position plot (`X` vs `Y`):
- Black line = ground truth trajectory
- Dashed colored lines = 5 samples drawn from the prior

The spread of sampled trajectories reflects model uncertainty. Tight clustering = low variance in the learned latent space; wide spread = higher uncertainty or a looser prior fit.

---

## Data format

CSV files follow the pattern `*_*_process_noise_real.csv` / `*_*_process_noise_simulated.csv`.

Each file is `[T × 13]` after loading (first column dropped, remainder transposed to `[12, T]`):
- Columns 0–8: state variables (including a time channel at index 0)
- Columns 9–11: control inputs

---

## Dependencies

```
torch
pandas
matplotlib
glob (stdlib)
```

No special install beyond PyTorch. GPU optional — both scripts fall back to CPU automatically.

---

## Notes

- The MI-VAE test only uses `decoderA` (real noise decoder). Swap in `decoderB` if you're evaluating on simulated trajectories.
- Normalization stats are saved in the checkpoint, so no need to recompute them at test time.
- Both decoders are autoregressive: prediction errors accumulate over long horizons. Short rollouts will be tighter than long ones.

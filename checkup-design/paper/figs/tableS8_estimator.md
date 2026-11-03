| item | setting |
|---|---|
| Training data | 1000 ageing states per chemistry, Latin-hypercube sampling; PyBaMM SPMe; split 70/15/15 by state |
| NMC model | O'Regan et al. LG M50T parameters, lumped thermal; C/10 and C/2 discharges to 3.0 V |
| NMC sampling ranges | LLI (extra) -0.06 to 0.25, LAM_PE -0.10 to 0.20, LAM_NE -0.05 to 0.35; D_n scaling 1-8, D_p 0.15-2 (log-uniform) |
| LFP model | Prada et al. A123 parameters, capacity 1.1 Ah, isothermal 50 C; C/25 and C/3 discharges to 2.5 V |
| LFP sampling ranges | LLI (extra) -0.05 to 0.50, LAM_PE -0.90 to 0, LAM_NE 0.05 to 0.65; D_n scaling 1-16, D_p 1-400 (log-uniform) |
| Windows | two per estimate, one per C-rate; size 15-100 % of nominal capacity, unknown start; 128 points; noise 2 mV, 0.5 C |
| Encoder | per window: Conv1d(2-32, k7), Conv1d(32-32, k5), max-pool, Conv1d(32-64, k5), max-pool, bidirectional LSTM (64); dense 128; GELU; dropout 0.1; 117 k parameters |
| Latent outputs | LLI, LAM_PE, LAM_NE, log D_n, log D_p through scaled sigmoids over the sampling ranges; window starts as sigmoid fractions of the feasible range |
| Decoder | electrode balancing; finite-volume particle diffusion (20 shells, 32 + 48 time steps, 3 Picard iterations for D(c)); Butler-Volmer kinetics; ohmic term; lumped electrolyte polarisation calibrated per C-rate on training curves |
| Training | label MSE + Gaussian voltage likelihood with learned variance; Adam, lr 1e-3, batch 256, 10 000 pairs per epoch, <= 60 epochs, early stopping (patience 12), gradient clipping at 5; seeds 0, 1, 2 |
| Refinement | 8 Levenberg-Marquardt iterations, central-difference Jacobian, Gaussian prior around the encoder output (s.d. 3 in logit units); NMC only |
| Evidence for reported modes | NMC: LAM_NE correlation with published reference 0.1-0.5, bias -10 to -15 pp (not used). LFP: refinement increased the error of LLI and LAM_NE for every seed; decoder misfit 12 mV on real vs 2 mV on simulated windows (encoder used). LFP LAM_PE: two-window CRB 9.5 pp vs label spread 26 pp (not identifiable) |

The estimator was trained only on simulations and used unchanged in every analysis of this paper.

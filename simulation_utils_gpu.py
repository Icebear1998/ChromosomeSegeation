# simulation_utils_gpu.py
"""
GPU-based simulations for time-varying chromosome segregation models.
*Note- this was made when the version I had only worked for time-varying-k models, so it will need updates

Alternative version of simulation_utils with a GPU version of `run_simulation_for_dataset`
Uses pytorch tensors to evolve a bunch of trajectories in parallel

It should mirror the logic of MultiMechanismSimulationTimevary (but would be worth double-checking):
- time-varying k(t) = min(k_1 * t, k_max)
- feedback / onion mechanisms via weights W_i
- inhomogeneous Poisson for event times
- burst sizes vs single-cohesin events

Make sure that you have the correct version of pytorch installed based on your hardware!
If no GPU backend is available, script auto runs on the CPU

Pytorch install instructions based on GPU------------------------------------

For all versions, first go to this website: https://pytorch.org/get-started/locally/
For "pytorch build", choose "stable"
Then follow instructions based on your GPU:

NVIDIA (RTX/GTX/Quadro/Tesla/Titan) (Anything with CUDA- so anything after the 17th century):
Install CUDA-enabled pytorch:
    1. For "compute platform", choose "CUDA" (ex. 12.6, 12.8, 13.0 - just use the default suggested)
    2. Copy/run the pip install command it shows
GPU device = "cuda"
    
AMD (Radeon/MI):
Install ROCm-enabled pytorch:
    1. For "compute platform", choose "ROCm" (ex. 6.8)
    2. Copy/run the pip install command it shows
GPU device: "hip"

Intel (Arc/Iris Xe/Data center GPUs):
requires pyyorch + intel extension for pyyorch + intel oneAPI runtime
A. Install pytorch (intel GPUs do use the CPU version of pytorch):
    1. For "compute platform", choose "CPU"
    2. Copy/run the pip install command it shows
B. Install Intel extension for pytorch (IPEX) (this puts it on the GPU):
    pip install intel-extension-for-pytorch
3. Make sure intel oneAPI GPU runtime is installed:
   - Windows: install latest Intel GPU driver (includes oneAPI Level Zero)
   - Linux: install packages: intel-opencl-icd, intel-level-zero-gpu
GPU device = "xpu"

To run this without a GPU:
Install CPU pytorch
    1. For "compute platform", choose "CPU"
    2. Copy/run the pip install command it shows
GPU device = "cpu"

"""

import numpy as np
import torch

# importing this from the original simulation_utils instead of re-defining (for continuity)
from simulation_utils import calculate_k1_from_params 

# Choose torch device- at the moment, only NVIDIA has been tested
if torch.cuda.is_available():
    device = torch.device("cuda") # NVIDIA
elif torch.backends.hip.is_available():
    device = torch.device("hip") # AMD
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    device = torch.device("xpu") # Intel
else:
    device = torch.device("cpu") # CPU fallback

print(f"Using device: {device}")


def _compute_onion_weights(initial_state_list, n_inner):
    """
    Compute onion feedback weights W_i from initial cohesin counts N_i and n_inner

    Should match the logic in _get_propensity_calculator/_calculate_effective_total_state
    """
    # New helper for onion weight calc
    N = np.array(initial_state_list, dtype=float)
    W = np.ones_like(N, dtype=float)
    for i, N_i in enumerate(N):
        if N_i > n_inner:
            W[i] = (N_i / n_inner) ** (-1.0 / 3.0)
        else:
            W[i] = 1.0
    return torch.tensor(W, dtype=torch.float32, device=device)


def _calculate_next_event_time_vector(current_time, total_effective_state,
                                      k_1_val, k_max_val):
    """
    Vector version of MultiMechanismSimulationTimevary _calculate_next_event_time_at_time

    Args:
        current_time (torch.Tensor): shape (S_active,)
        total_effective_state (torch.Tensor): shape (S_active,)
        k_1_val (float)
        k_max_val (float or np.inf)

    Returns:
        torch.Tensor: tau for each active trajectory (shape (S_active,))
    """
    # default tau = tiny positive step
    tau = torch.full_like(current_time, 1e-10)

    # mask- only update where time > 0 and effective_state > 0
    mask = (current_time > 0.0) & (total_effective_state > 0.0)
    if not torch.any(mask):
        return tau

    ct = current_time[mask]
    tes = total_effective_state[mask]

    # scalar constants on CPU side
    k_1 = float(k_1_val)
    k_max = float(k_max_val)

    # handle k_max = inf (linear phase) vs finite k_max (linear to constant)
    if np.isinf(k_max):
        # pure linear phase: k(t) = k_1 * t
        r_1 = torch.rand_like(ct)
        # c = 2 * log(r_1) / (k_1 * tes)
        c = 2.0 * torch.log(r_1) / (k_1 * tes)
        b = 2.0 * ct
        discriminant = b * b - 4.0 * c
        # clamp discriminant to avoid negatives from noise
        discriminant = torch.clamp(discriminant, min=0.0)
        tau_lin = (-b + torch.sqrt(discriminant)) / 2.0

        # where tau_lin <= 0, keep default tau- update otherwise
        valid = tau_lin > 0.0
        tau[mask] = torch.where(valid, tau_lin, tau[mask])
        return tau

    # mixed case- linear phase up to t_max, constant k_max after
    t_max = k_max / k_1

    # decide which trajectories are already in constant phase
    in_const = ct >= t_max
    in_linear = ~in_const

    # constant phase- exponential with rate = k_max * total_effective_state
    if torch.any(in_const):
        ct_const = ct[in_const]
        tes_const = tes[in_const]
        # rate = k_max * tes
        rate_const = k_max * tes_const
        rate_const = torch.clamp(rate_const, min=1e-12)
        exp_samples = torch.empty_like(rate_const).exponential_()
        tau_const = exp_samples / rate_const
        tau_subset = tau[mask]
        tau_subset[in_const] = tau_const
        tau[mask] = tau_subset

    # linear phase- k(t) = k_1 * t, but may cross t_max
    if torch.any(in_linear):
        ct_lin = ct[in_linear]
        tes_lin = tes[in_linear]

        r_1 = torch.rand_like(ct_lin)
        c = 2.0 * torch.log(r_1) / (k_1 * tes_lin)
        b = 2.0 * ct_lin
        discriminant = b * b - 4.0 * c
        discriminant = torch.clamp(discriminant, min=0.0)
        tau_lin = (-b + torch.sqrt(discriminant)) / 2.0

        # if tau_lin <= 0, fallback to tiny step
        valid_lin = tau_lin > 0.0
        tau_lin = torch.where(valid_lin, tau_lin, torch.full_like(tau_lin, 1e-10))

        # check t_max crossing
        # if current_time + tau_lin > t_max, go to t_max
        cross_mask = (ct_lin + tau_lin) > t_max
        tau_lin_clamped = torch.where(
            cross_mask,
            torch.tensor(t_max, device=ct_lin.device) - ct_lin,
            tau_lin
        )

        tau_subset = tau[mask]
        tau_subset[in_linear] = tau_lin_clamped
        tau[mask] = tau_subset

    return tau


def run_simulation_for_dataset_gpu(mechanism, params, n0_list,
                                   num_simulations=500, max_time=1000.0,
                                   rng_seed=None):
    """
    GPU version of run_simulation_for_dataset.

    Args:
        mechanism (str): name of model (currently for time-varying models)
        params (dict): {
            'n1','n2','n3','N1','N2','N3','k_1' or ('k_max' and 'tau'), optional 'burst_size','n_inner'
        }
        n0_list (list): [n1_threshold, n2_threshold, n3_threshold]
        num_simulations (int): # of trajectories to run
        max_time (float): max simulation time (mostly to cap iterations)
        rng_seed (int or None): for reproducibility

    Returns:
        (delta_t12_list, delta_t32_list) or (None, None) on complete failure
    """
    # GPU version works in place of run_simulation_for_datset from simultion_utils.py
    # old version called run_simulation_for_dataset in a loop- this version is GPU vectorized and does not call MultiMechanismTimevary
    if rng_seed is not None:
        torch.manual_seed(rng_seed)

    # 1- setup initial states and parameters on GPU ---------------------------
    # should be the same logic as old initial_state = [N1, N2, N3]- just a torch tensor
    initial_state = torch.tensor(
        [int(round(params['N1'])), int(round(params['N2'])), int(round(params['N3']))],
        dtype=torch.float32,
        device=device
    )  # shape (3,)
    
    # n0 was prev a list of thresholds- now a GPU tensor
    n0 = torch.tensor(n0_list, dtype=torch.float32, device=device)  # shape (3,)

    # k_1 from params (handles k_max / tau or explicit k_1)
    # Handle simple mechanism (constant k) vs time-varying
    if 'k' in params:
        # Simple mechanism: k(t) = k (constant)
        # We map this to the time-varying model by setting k_max = k and k_1 = very large
        # This ensures k(t) = min(k_1 * t, k_max) becomes k_max almost instantly
        k_max_val = float(params['k'])
        k_1_val = 1e20  # Effectively infinite slope
    else:
        # Time-varying mechanism
        k_1_val = calculate_k1_from_params(params)
        k_max_val = float(params.get('k_max', float('inf')))

    k_1 = torch.tensor(k_1_val, dtype=torch.float32, device=device)
    k_max = torch.tensor(k_max_val, dtype=torch.float32, device=device)

    num_chromosomes = initial_state.shape[0]  # should be 3

    # following mechanism-specific stuff is moved here instead of inside MultiMechanismTimevary
    # mechanism type flags
    onion_mechanisms = ['time_varying_k_feedback_onion', 'time_varying_k_combined']
    burst_mechanisms = ['time_varying_k_fixed_burst', 'time_varying_k_burst_onion', 'time_varying_k_combined']

    # burst size
    if mechanism in burst_mechanisms:
        burst_size_val = float(params['burst_size'])
    else:
        burst_size_val = 1.0  # single cohesin
    burst_size = torch.tensor(burst_size_val, dtype=torch.float32, device=device)

    # onion weights
    if mechanism in onion_mechanisms:
        n_inner_val = float(params['n_inner'])
        W = _compute_onion_weights(initial_state.cpu().numpy(), n_inner_val)  # shape (3,)
    else:
        W = torch.ones(num_chromosomes, dtype=torch.float32, device=device)

    # 2- batch initialization over trajectories -------------------------------
    # state: (S, 3)
    # replicate initial state for all trajectories on GPU
    state = initial_state.unsqueeze(0).repeat(num_simulations, 1).clone()
    # times: (S,)
    times = torch.zeros(num_simulations, dtype=torch.float32, device=device)
    # sep_times: (S, 3), initial -1 ("None")
    sep_times = -torch.ones(num_simulations, num_chromosomes, dtype=torch.float32, device=device)

    # active trajectories (still evolving)
    active = torch.ones(num_simulations, dtype=torch.bool, device=device)

    # rough upper bound on iterations to avoid infinite loops
    # moved this here from MultiMechanismSimulationTimevary
    max_steps = 100000

    # 3- main simulation loop (vectorized across trajectories) ----------------
    # MAJOR CHANGE: single loop over steps, updates many trajectories in parallel with torch
    # old version looped over each trajectory
    for step in range(max_steps):
        if not torch.any(active):
            break

        # extract active subsets
        idx = active.nonzero(as_tuple=False).squeeze(-1)
        state_a = state[idx] # (Sa, 3)
        times_a = times[idx] # (Sa,)
        sep_a = sep_times[idx] # (Sa, 3)

        # check which trajectories still have unseparated chromosmoes
        still_unseparated = torch.any(sep_a < 0.0, dim=1)

        if not torch.any(still_unseparated):
            # nothing left to simulate :)
            active[idx] = False
            break

        # safety- only update unseparated chroms
        idx2 = idx[still_unseparated]
        state_a = state[idx2]   # (Sb, 3)
        times_a = times[idx2]
        sep_a = sep_times[idx2]

        # get effective total state (should match _calculate_effective_total_state)
        # states clamped at >=0
        state_nonneg = torch.clamp(state_a, min=0.0)
        if mechanism in onion_mechanisms:
            total_effective_state = (state_nonneg * W.unsqueeze(0)).sum(dim=1)  # (Sb,)
        else:
            total_effective_state = state_nonneg.sum(dim=1)  # (Sb,)

        # if effective state = zero, nothing else happens- mark as finished
        zero_eff = total_effective_state <= 0.0
        if torch.any(zero_eff):
            # set any remaining sep_times to current time
            idx_zero = idx2[zero_eff]
            sep_z = sep_times[idx_zero]
            t_z = times[idx_zero]
            mask_unsep_z = sep_z < 0.0
            sep_z[mask_unsep_z] = t_z.unsqueeze(1).expand_as(sep_z)[mask_unsep_z]
            sep_times[idx_zero] = sep_z
            active[idx_zero] = False

        # continue with any that still have positive effective state
        pos_eff_mask = total_effective_state > 0.0
        if not torch.any(pos_eff_mask):
            continue

        idx3 = idx2[pos_eff_mask]
        state_b = state[idx3] # (Sc, 3)
        times_b = times[idx3] # (Sc,)
        sep_b = sep_times[idx3]
        tes_b = total_effective_state[pos_eff_mask]

        # 3A- calculate time to next event using inhomogeneous poisson --------
        # More direct call to vectorized inhomogenous poisson time calculator
        tau_b = _calculate_next_event_time_vector(times_b, tes_b, k_1_val, k_max_val)
        # update times
        new_times_b = times_b + tau_b

        # 3B- recalculate k(t) @ new times for reaction selection -------------
        k_linear = k_1 * new_times_b
        if np.isinf(k_max_val):
            k_t_b = torch.clamp(k_linear, min=1e-10)
        else:
            k_t_b = torch.clamp(torch.minimum(k_linear, k_max), min=1e-10)

        # 3C- calculate chromosome propensities -------------------------------
        state_nonneg_b = torch.clamp(state_b, min=0.0)  # (Sc, 3)
        if mechanism in onion_mechanisms:
            propensities = k_t_b.unsqueeze(1) * state_nonneg_b * W.unsqueeze(0) # (Sc, 3)
        else:
            propensities = k_t_b.unsqueeze(1) * state_nonneg_b # (Sc, 3)

        total_propensity = propensities.sum(dim=1) # (Sc,)

        # if total_propensity <= 0, nothing else happens- finalize sep_times
        zero_prop = total_propensity <= 0.0
        if torch.any(zero_prop):
            idx_zero = idx3[zero_prop]
            sep_z = sep_times[idx_zero]
            t_z = new_times_b[zero_prop]
            mask_unsep_z = sep_z < 0.0
            sep_z[mask_unsep_z] = t_z.unsqueeze(1).expand_as(sep_z)[mask_unsep_z]
            sep_times[idx_zero] = sep_z
            active[idx_zero] = False

        # continue if positive propensity
        pos_prop_mask = total_propensity > 0.0
        if not torch.any(pos_prop_mask):
            continue

        idx4 = idx3[pos_prop_mask]
        state_c = state[idx4] # (Sd, 3)
        times_c = new_times_b[pos_prop_mask] # (Sd,)
        sep_c = sep_times[idx4]
        prop_c = propensities[pos_prop_mask] # (Sd, 3)
        total_prop_c = total_propensity[pos_prop_mask]

        # 3D- which chromosome degrades for each trajectory -------------------
        # rxn selection done by cumulative propensities in torch instead of MultiMechanismTimevary
        # draw uniform r in [0, total_prop_c)
        r = torch.rand_like(total_prop_c) * total_prop_c # (Sd,)
        cum_prop = torch.cumsum(prop_c, dim=1) # (Sd, 3)

        # compare cum_prop > r, take first index where this is true
        # I really could've gone with a better variable name here
        # mask shape: (Sd, 3)
        comp = cum_prop > r.unsqueeze(1)
        # for each row, argmax of comp gives first True (assuming at least one True)
        selected_chrom = comp.float().argmax(dim=1) # (Sd,)

        # 3E- apply degradation (burst size or single-cohesin) ----------------
        # convert integer indices and index into state_c
        S_d = state_c.shape[0]
        row_idx = torch.arange(S_d, device=device)
        # subtract burst_size (but not below 0)
        new_state_c = state_c.clone()
        new_state_c[row_idx, selected_chrom] = torch.clamp(
            new_state_c[row_idx, selected_chrom] - burst_size, min=0.0
        )

        # 3F- check for chromosome separation thresholds ----------------------
        # sep_c: (Sd, 3), times_c: (Sd,), new_state_c: (Sd, 3)
        for chr_idx in range(num_chromosomes):
            # if sep time not set yet, and state <= n0, set sep time to times_c
            not_yet = sep_c[:, chr_idx] < 0.0
            crossed = new_state_c[:, chr_idx] <= n0[chr_idx]
            newly_sep = not_yet & crossed
            if torch.any(newly_sep):
                sep_c[newly_sep, chr_idx] = times_c[newly_sep]

        # write results back into global tensors
        state[idx4] = new_state_c
        times[idx4] = times_c
        sep_times[idx4] = sep_c

        # deactivate trajectories if they have all sep_times set
        fully_sep = torch.all(sep_c >= 0.0, dim=1)
        if torch.any(fully_sep):
            active[idx4[fully_sep]] = False

        # probably optional- safety break if times exceed max_time too badly
        if torch.max(times) > max_time * 5:
            break

    # 4- finalize- any sep_times still < 0, set to current time ---------------
    unsep_mask = sep_times < 0.0
    if torch.any(unsep_mask):
        # for each trajectory, fill remaining with final time
        final_times = times.unsqueeze(1).expand_as(sep_times)
        sep_times[unsep_mask] = final_times[unsep_mask]

    # if all simulations failed, do same thing old versio does for CPU failure
    if torch.any(torch.isnan(sep_times)):
        return None, None

    # extract delta t12 and t32
    T1 = sep_times[:, 0]
    T2 = sep_times[:, 1]
    T3 = sep_times[:, 2]

    delta_t12 = (T1 - T2).detach().cpu().numpy()
    delta_t32 = (T3 - T2).detach().cpu().numpy()

    # if those are empty, return failure (like CPU version does)
    if delta_t12.size == 0 or delta_t32.size == 0:
        return None, None

    return delta_t12, delta_t32

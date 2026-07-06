import json
import numpy as np
import matplotlib.pyplot as plt

def fit_and_plot_laws():
    # 1. Load the generated sweep results
    try:
        with open("sweep_results.json", "r") as f:
            results = json.load(f)
    except FileNotFoundError:
        print("Error: sweep_results.json not found! Please wait for train_scaling.py to finish running completely.")
        return

    # Extract Parameter Sweep Data
    param_data = results["parameter_sweep"]
    names = [d["name"] for d in param_data]
    N_vals = np.array([d["N"] for d in param_data], dtype=float)
    L_N_vals = np.array([d["loss"] for d in param_data], dtype=float)

    # Extract Data Sweep Data
    data_sweep = results["data_sweep"]
    D_vals = np.array([d["D"] for d in data_sweep], dtype=float)
    L_D_vals = np.array([d["loss"] for d in data_sweep], dtype=float)

    # 2. Fit power laws in log-log space using numpy.polyfit
    # log(L(N)) = constant - alpha_N * log(N) -> slope is -alpha_N
    slope_N, intercept_N = np.polyfit(np.log10(N_vals), np.log10(L_N_vals), 1)
    alpha_N = -slope_N

    # log(L(D)) = constant - alpha_D * log(D) -> slope is -alpha_D
    slope_D, intercept_D = np.polyfit(np.log10(D_vals), np.log10(L_D_vals), 1)
    alpha_D = -slope_D

    # Calculate scaling ratio gamma
    gamma = alpha_N / alpha_D

    print("EMPIRICAL SCALING LAW RESULTS")
    print(f"Calculated alpha_N (Parameter exponent): {alpha_N:.4f}")
    print(f"Calculated alpha_D (Data exponent):      {alpha_D:.4f}")
    print(f"Calculated Scaling Ratio (gamma):        {gamma:.4f}")

    # 3. Generate Dual Log-Log Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left Plot: Parameter Scaling
    ax1.scatter(N_vals, L_N_vals, color='red', s=100, zorder=5, label='Empirical Models')
    for i, txt in enumerate(names):
        ax1.annotate(f" {txt}", (N_vals[i], L_N_vals[i]), fontsize=11, weight='bold')
    
    # Fit line projection
    N_fit_line = np.linspace(min(N_vals)*0.5, max(N_vals)*2, 100)
    L_N_fit_line = 10**(intercept_N) * (N_fit_line**slope_N)
    ax1.plot(N_fit_line, L_N_fit_line, 'k--', alpha=0.7, label=f'Fit Line ($\\alpha_N$ = {alpha_N:.3f})')
    
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Non-Embedding Parameters ($N$)', fontsize=12)
    ax1.set_ylabel('Minimal Validation Cross-Entropy Loss ($L$)', fontsize=12)
    ax1.set_title('Parameter Scaling Law Fit', fontsize=13, weight='bold')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend(fontsize=10)

    # Right Plot: Data Scaling
    ax2.scatter(D_vals, L_D_vals, color='blue', s=100, zorder=5, label='Data Budgets')
    fractions = ["10%", "25%", "50%", "100%"]
    for i, txt in enumerate(fractions):
        ax2.annotate(f" {txt}", (D_vals[i], L_D_vals[i]), fontsize=11, weight='bold')
        
    # Fit line projection
    D_fit_line = np.linspace(min(D_vals)*0.5, max(D_vals)*2, 100)
    L_D_fit_line = 10**(intercept_D) * (D_fit_line**slope_D)
    ax2.plot(D_fit_line, L_D_fit_line, 'k--', alpha=0.7, label=f'Fit Line ($\\alpha_D$ = {alpha_D:.3f})')
    
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Dataset Size in Total Tokens ($D$)', fontsize=12)
    ax2.set_ylabel('Minimal Validation Cross-Entropy Loss ($L$)', fontsize=12)
    ax2.set_title('Data Scaling Law Fit', fontsize=13, weight='bold')
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend(fontsize=10)

    plt.suptitle(f"Neural Language Model Power-Law Scaling Metrics ($\\gamma$ = {gamma:.3f})", fontsize=15, weight='bold', y=1.02)
    plt.tight_layout()
    
    # Save chart asset cleanly to environment workspace
    plt.savefig("scaling_laws.png", dpi=300, bbox_inches='tight')
    plt.show()
    print("Success! High-resolution scaling curves exported to 'scaling_laws.png'")

if __name__ == "__main__":
    fit_and_plot_laws()

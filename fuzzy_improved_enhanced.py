import numpy as np
import pandas as pd
import glob, os
from scipy.fft import fft
from scipy import stats
from scipy.signal import welch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import skfuzzy as fuzz

import warnings
warnings.filterwarnings('ignore')

def window_signal(signal, window_size=512, step=256):
    """
    Splits signal into overlapping windows.
    Increased window size to 512 for better frequency resolution.
    """
    windows = []
    for i in range(0, len(signal) - window_size + 1, step):
        window = signal[i:i + window_size]
        # Apply Hamming window to reduce spectral leakage
        window = window * np.hamming(len(window))
        windows.append(window)
    return np.array(windows)

def extract_features(csv_path, fs=1000):
    """
    Extract comprehensive time-domain and frequency-domain features.
    Enhanced with additional discriminative features.
    """
    df = pd.read_csv(csv_path)
    vib = df['Vibration'].values
    
    windows = window_signal(vib, window_size=512, step=256)
    features = []
    
    for w in windows:
        N = len(w)
        
        # ============ TIME DOMAIN FEATURES ============
        # Statistical moments
        mean = np.mean(w)
        std = np.std(w)
        variance = np.var(w)
        rms = np.sqrt(np.mean(w**2))
        
        # Shape indicators
        skewness = stats.skew(w)
        kurtosis = stats.kurtosis(w)
        
        # Amplitude features
        peak = np.max(np.abs(w))
        peak_to_peak = np.ptp(w)
        crest_factor = peak / (rms + 1e-10)
        
        # Shape factor and impulse factor
        shape_factor = rms / (np.mean(np.abs(w)) + 1e-10)
        impulse_factor = peak / (np.mean(np.abs(w)) + 1e-10)
        
        # Clearance factor
        clearance_factor = peak / ((np.mean(np.sqrt(np.abs(w))))**2 + 1e-10)
        
        # Energy
        energy = np.sum(w**2)
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(w)) != 0)
        
        # NEW: Waveform factor
        waveform_factor = rms / (np.mean(np.abs(w)) + 1e-10)
        
        # NEW: Mean absolute deviation
        mad = np.mean(np.abs(w - mean))
        
        # NEW: Interquartile range
        iqr = np.percentile(w, 75) - np.percentile(w, 25)
        
        # NEW: Entropy (measure of signal complexity)
        hist, _ = np.histogram(w, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zeros
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # ============ FREQUENCY DOMAIN FEATURES ============
        # FFT computation
        fft_vals = np.abs(fft(w))
        freqs = np.fft.fftfreq(N, 1/fs)
        
        # Use only positive frequencies
        pos_mask = freqs >= 0
        freqs = freqs[pos_mask]
        fft_vals = fft_vals[pos_mask]
        
        # Normalize spectrum
        fft_vals_norm = fft_vals / (np.sum(fft_vals) + 1e-10)
        
        # Dominant frequency
        dom_freq = freqs[np.argmax(fft_vals)]
        
        # Spectral centroid (frequency center of mass)
        spectral_centroid = np.sum(freqs * fft_vals_norm)
        
        # Spectral spread (standard deviation of spectrum)
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * fft_vals_norm))
        
        # Spectral rolloff (frequency below which 85% of energy is contained)
        cumsum = np.cumsum(fft_vals_norm)
        spectral_rolloff = freqs[np.where(cumsum >= 0.85)[0][0]] if np.any(cumsum >= 0.85) else freqs[-1]
        
        # Spectral flatness (measure of noise vs tones)
        geometric_mean = np.exp(np.mean(np.log(fft_vals + 1e-10)))
        arithmetic_mean = np.mean(fft_vals)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # NEW: Spectral entropy
        spectral_entropy = -np.sum(fft_vals_norm * np.log2(fft_vals_norm + 1e-10))
        
        # Energy in frequency bands (more granular)
        # Ultra-low frequency (0-5 Hz)
        ultra_low_energy = np.sum(fft_vals[(freqs >= 0) & (freqs <= 5)]**2)
        
        # Very low frequency (5-20 Hz)
        vlow_energy = np.sum(fft_vals[(freqs > 5) & (freqs <= 20)]**2)
        
        # Low frequency (20-60 Hz) - bearing faults
        low_energy = np.sum(fft_vals[(freqs > 20) & (freqs <= 60)]**2)
        
        # Mid-low frequency (60-120 Hz)
        mid_low_energy = np.sum(fft_vals[(freqs > 60) & (freqs <= 120)]**2)
        
        # Mid frequency (120-200 Hz) - gear mesh
        mid_energy = np.sum(fft_vals[(freqs > 120) & (freqs <= 200)]**2)
        
        # Mid-high frequency (200-350 Hz)
        mid_high_energy = np.sum(fft_vals[(freqs > 200) & (freqs <= 350)]**2)
        
        # High frequency (350-500 Hz)
        high_energy = np.sum(fft_vals[(freqs > 350) & (freqs <= 500)]**2)
        
        # Total spectral energy
        total_spectral_energy = np.sum(fft_vals**2)
        
        # Band energy ratios (discriminative for fault types)
        low_to_high_ratio = low_energy / (high_energy + 1e-10)
        mid_to_total_ratio = mid_energy / (total_spectral_energy + 1e-10)
        low_to_total_ratio = low_energy / (total_spectral_energy + 1e-10)
        high_to_total_ratio = high_energy / (total_spectral_energy + 1e-10)
        
        # NEW: Harmonic ratio (sum of first 5 harmonics vs total)
        if dom_freq > 0:
            harmonic_energy = 0
            for h in range(1, 6):
                harm_freq = dom_freq * h
                harm_idx = np.argmin(np.abs(freqs - harm_freq))
                harmonic_energy += fft_vals[harm_idx]**2
            harmonic_ratio = harmonic_energy / (total_spectral_energy + 1e-10)
        else:
            harmonic_ratio = 0
        
        # Power Spectral Density features using Welch's method
        f_welch, psd = welch(w, fs=fs, nperseg=min(256, len(w)))
        psd_peak = np.max(psd)
        psd_mean = np.mean(psd)
        psd_std = np.std(psd)
        psd_median = np.median(psd)
        
        # Compile all features (38 features total)
        features.append([
            # Time domain (18 features)
            mean, std, variance, rms, skewness, kurtosis,
            peak, peak_to_peak, crest_factor, shape_factor,
            impulse_factor, clearance_factor, energy, zero_crossings,
            waveform_factor, mad, iqr, entropy,
            
            # Frequency domain (20 features)
            dom_freq, spectral_centroid, spectral_spread, spectral_rolloff,
            spectral_flatness, spectral_entropy,
            ultra_low_energy, vlow_energy, low_energy, mid_low_energy,
            mid_energy, mid_high_energy, high_energy, total_spectral_energy,
            low_to_high_ratio, mid_to_total_ratio, low_to_total_ratio,
            high_to_total_ratio, harmonic_ratio,
            psd_peak, psd_mean, psd_std, psd_median
        ])
    
    return np.array(features, dtype=np.float64)

# Load data
print("="*70)
print("LOADING AND PROCESSING DATA")
print("="*70)

X, y = [], []

csv_files = sorted(glob.glob("./*.csv"))

if len(csv_files) == 0:
    print("Warning: No CSV files found in current directory")
else:
    for file in csv_files:
        fault_name = os.path.basename(file).replace(".csv", "").replace("_", " ").title()
        
        try:
            feats = extract_features(file)
            X.extend(feats)
            y.extend([fault_name] * len(feats))
            print(f"Processed {fault_name}: {len(feats)} samples")
        except Exception as e:
            print(f"Error processing {file}: {e}")

X = np.array(X, dtype=np.float64)
y = np.array(y)

print(f"\nDataset shape: {X.shape}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of classes: {len(np.unique(y))}")

# Encode labels
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Handle any NaN or infinite values
X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)

print("\n" + "="*70)
print("FEATURE SELECTION AND PREPROCESSING")
print("="*70)

# Feature selection - select top k most discriminative features
selector = SelectKBest(f_classif, k=min(25, X.shape[1]))
X_selected = selector.fit_transform(X, y_enc)

selected_features = selector.get_support(indices=True)
print(f"Selected {X_selected.shape[1]} most discriminative features")

# Use StandardScaler for better FCM performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

print("Fault classes:")
for i, c in enumerate(encoder.classes_):
    count = np.sum(y_enc == i)
    print(f"  {i} → {c} ({count} samples)")

print(f"\nFeature statistics after scaling:")
print(f"  Mean: {np.mean(X_scaled):.6f}")
print(f"  Std: {np.std(X_scaled):.6f}")
print(f"  Min: {np.min(X_scaled):.6f}")
print(f"  Max: {np.max(X_scaled):.6f}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc,
    test_size=0.20,  # 80-20 split for more training data
    random_state=42,
    stratify=y_enc
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Initialize cluster centers using class means (better initialization)
n_classes = len(np.unique(y_train))
init_centers = np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])

print("\n" + "="*70)
print("TRAINING FUZZY C-MEANS CLASSIFIER")
print("="*70)
print(f"  Number of clusters: {n_classes}")
print(f"  Number of features: {X_train.shape[1]}")
print(f"  Number of training samples: {X_train.shape[0]}")
print(f"  Fuzziness parameter (m): 2.0")
print(f"  Convergence threshold: 0.00001")
print(f"  Max iterations: 3000")
print(f"  Initialization: Class mean centers")

# Create initial membership matrix from class means
# Convert class means to membership matrix format
init_u = np.zeros((n_classes, X_train.shape[0]))
for i in range(n_classes):
    init_u[i, y_train == i] = 1.0

# Train Fuzzy C-Means with optimized parameters
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    X_train.T,  # Data should be (n_features, n_samples)
    c=n_classes,
    m=2.0,  # Increased fuzziness for smoother boundaries
    error=0.00001,  # Tighter convergence
    maxiter=3000,  # More iterations
    init=init_u  # Initialize with membership matrix
)

print(f"\nTraining completed in {p} iterations")
print(f"Fuzzy partition coefficient (FPC): {fpc:.6f}")
print(f"  (FPC close to 1.0 indicates well-separated clusters)")

def fuzzy_predict_enhanced(X, centers, m=2.0):
    """
    Enhanced fuzzy prediction using membership degrees.
    """
    n_samples = X.shape[0]
    n_centers = centers.shape[0]
    
    # Calculate membership values for all samples
    u_pred = np.zeros((n_centers, n_samples))
    
    for i in range(n_samples):
        # Calculate distances to all centers
        distances = np.linalg.norm(centers - X[i], axis=1)
        
        # Avoid division by zero
        distances = np.maximum(distances, 1e-10)
        
        # Calculate membership values using fuzzy c-means formula
        for j in range(n_centers):
            sum_term = np.sum((distances[j] / distances) ** (2 / (m - 1)))
            u_pred[j, i] = 1.0 / sum_term
    
    # Assign to cluster with highest membership
    predictions = np.argmax(u_pred, axis=0)
    
    return predictions, u_pred

# Predict on training set
y_train_pred, u_train = fuzzy_predict_enhanced(X_train, cntr, m=2.0)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Predict on test set
y_pred, u_test = fuzzy_predict_enhanced(X_test, cntr, m=2.0)
test_accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*70)
print("CLASSIFICATION PERFORMANCE")
print("="*70)
print(f"Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Testing Accuracy:  {test_accuracy*100:.2f}%")
print("="*70)

print("\nDetailed Classification Report (Test Set):\n")
print(classification_report(
    y_test,
    y_pred,
    target_names=encoder.classes_,
    digits=4
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    ax=ax1,
    cbar_kws={'label': 'Count'}
)
ax1.set_xlabel("Predicted Fault", fontsize=12)
ax1.set_ylabel("Actual Fault", fontsize=12)
ax1.set_title(f"Confusion Matrix (Counts)\nTest Accuracy: {test_accuracy*100:.2f}%", fontsize=14, fontweight='bold')

# Percentages
sns.heatmap(
    cm_percent,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    xticklabels=encoder.classes_,
    yticklabels=encoder.classes_,
    ax=ax2,
    cbar_kws={'label': 'Percentage (%)'}
)
ax2.set_xlabel("Predicted Fault", fontsize=12)
ax2.set_ylabel("Actual Fault", fontsize=12)
ax2.set_title("Confusion Matrix (Percentages)", fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\nConfusion matrix saved as 'confusion_matrix.png'")

# Cross-validation
print("\n" + "="*70)
print("PERFORMING 5-FOLD CROSS-VALIDATION")
print("="*70)

cv_scores = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y_enc), 1):
    X_cv_train, X_cv_val = X_scaled[train_idx], X_scaled[val_idx]
    y_cv_train, y_cv_val = y_enc[train_idx], y_enc[val_idx]
    
    # Initialize with membership matrix for this fold
    init_u_cv = np.zeros((n_classes, X_cv_train.shape[0]))
    for i in range(n_classes):
        init_u_cv[i, y_cv_train == i] = 1.0
    
    # Train FCM
    cntr_cv, _, _, _, _, _, _ = fuzz.cluster.cmeans(
        X_cv_train.T,
        c=n_classes,
        m=2.0,
        error=0.00001,
        maxiter=3000,
        init=init_u_cv
    )
    
    # Predict
    y_cv_pred, _ = fuzzy_predict_enhanced(X_cv_val, cntr_cv, m=2.0)
    score = accuracy_score(y_cv_val, y_cv_pred)
    cv_scores.append(score)
    print(f"  Fold {fold}: {score*100:.2f}%")

print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)
print(f"Mean CV Accuracy: {np.mean(cv_scores)*100:.2f}% (+/- {np.std(cv_scores)*100:.2f}%)")
print(f"Min CV Accuracy:  {np.min(cv_scores)*100:.2f}%")
print(f"Max CV Accuracy:  {np.max(cv_scores)*100:.2f}%")
print("="*70)

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Cross-Validation Mean: {np.mean(cv_scores)*100:.2f}%")
print("="*70)

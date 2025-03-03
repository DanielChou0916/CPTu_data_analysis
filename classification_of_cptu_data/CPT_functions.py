import numpy as np
import pandas as pd
def unit_weight_cal(sleeve_fric):
    gamat=12+1.5*np.log(sleeve_fric+0.1)
    return gamat

def total_stress_cal(unit_weight, depth):
    unit_weight = np.asarray(unit_weight, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)

    # Ensure no NaN values in inputs
    unit_weight = np.nan_to_num(unit_weight, nan=0)
    depth = np.nan_to_num(depth, nan=0)

    # Ensure depth is strictly increasing
    if np.any(np.diff(depth) <= 0):
        print("⚠️ Warning: Depth values are not strictly increasing! Fixing the issue.")
        sorted_indices = np.argsort(depth)
        depth = depth[sorted_indices]
        unit_weight = unit_weight[sorted_indices]

    # Compute layer thickness
    layer_thickness = np.diff(depth, prepend=0)

    # Compute cumulative stress
    sigma_v0 = np.cumsum(layer_thickness * unit_weight)

    # Debugging: Check for NaNs in the output
    if np.any(np.isnan(sigma_v0)):
        nan_idx = np.where(np.isnan(sigma_v0))[0][0]  # Get first NaN occurrence
        print(f"⚠️ NaN detected at index {nan_idx}:")
        print(f"   Thickness: {layer_thickness[nan_idx]}")
        print(f"   Previous Stress: {sigma_v0[nan_idx-1] if nan_idx > 0 else 'N/A'}")
        print(f"   Unit Weight: {unit_weight[nan_idx]}")
        print(f"   Depth: {depth[nan_idx]}")

    return sigma_v0


def u0_cal(depth, GWT=0):
    depth = np.asarray(depth)  # Ensure depth is a NumPy array

    # Compute pore water pressure based on depth relative to GWT
    u0 = np.where(depth <= GWT, 0, (depth - GWT) * 9.81)
    
    return u0

def sigma_eff_cal(sigma_v0, u0):
    sigma_v0 = np.asarray(sigma_v0)  # Convert to NumPy array
    u0 = np.asarray(u0)  # Convert to NumPy array

    # Compute effective stress directly using vectorized subtraction
    sigma_eff = sigma_v0 - u0

    return sigma_eff


# Function for CPT analysis parameters
def qnet_deltau(qt, sigma_v0, u0, u2):
    """
    Compute net cone resistance (q_net) and porewater pressure difference (delta_u).
    
    q_net = qt - sigma_v0
    delta_u = u2 - u0
    """
    q_net = qt - sigma_v0
    delta_u = u2 - u0
    return q_net, delta_u


# Function for normalized piezocone parameters
def Normalized_piez_paras(q_net, sigma_eff, fs, delta_u):
    """
    Compute normalized CPT parameters:
    - Normalized tip resistance: Q = q_net / sigma_eff
    - Normalized sleeve friction: F = 100 * fs / q_net
    - Normalized porewater pressure: Bq = delta_u / q_net (Bq < 0 is set to 0)
    """
    # Avoid division by zero errors
    sigma_eff = np.where(sigma_eff == 0, np.nan, sigma_eff)
    q_net = np.where(q_net == 0, np.nan, q_net)

    Q = q_net / sigma_eff
    F = 100 * fs / q_net
    F = np.clip(F, 0.01, 100)
    Bq = np.maximum(delta_u / q_net, 0)  # If Bq < 0, set to 0

    return Q, F, Bq


# CPT Material Index (Robertson & Wride, 1998)
def I_c_calculation(Q, F):
    """
    計算 CPT 材料指數 I_c
    Robertson & Wride (1998):
    I_c = sqrt( (3.47 - log10(Q))^2 + (1.22 + log10(F))^2 )
    """

    # 確保 Q 和 F > 0，避免 log10 出錯
    Q = np.where(Q > 0, Q, np.nan)
    F = np.where(F > 0, F, np.nan)

    log_Q = np.log10(Q)
    log_F = np.log10(F)

    I_c = np.sqrt((3.47 - log_Q) ** 2 + (1.22 + log_F) ** 2)
    return I_c


# Function for Qtn_n based on Ic and sigma_eff
def Qtn_n(I_c, sigma_eff, q_net):
    """
    Compute normalized cone resistance (Qtn) using Robertson's equation:
    - n = 0.381 * Ic + 0.05 * (sigma_eff / 101.325) - 0.15
    - Qtn = (q_net / 101.325) * (101.325 / sigma_eff) ^ n
    """
    n = 0.381 * I_c + 0.05 * (sigma_eff / 101.325) - 0.15
    Qtn = (q_net / 101.325) * (101.325 / sigma_eff) ** n

    return n, Qtn


# Function for material index Ic_m based on Qtn and F
def Ic_m_calculation(Qtn, F):
    """
    Compute the CPT Material Index (Ic_m) using Qtn and F.
    Ic_m = sqrt( (3.47 - log10(Qtn))² + (1.22 + log10(F))² )
    """
    Qtn = np.where(Qtn > 0, Qtn, np.nan)
    F = np.where(F > 0, F, np.nan)
    log_Qtn = np.log10(Qtn)
    log_F = np.log10(F)

    Ic_m = np.sqrt((3.47 - log_Qtn) ** 2 + (1.22 + log_F) ** 2)
    return Ic_m

#Function of soil classification based on Ic_m
#According to Robertson, 1990, 1991; 2009 Canadian Geotechnical Journal
# if wanna consider Zone 8,9 and 1, re-design use while loop with index rather than for loop.
def classification_soil(Ic_m, F, Qtn):
    #Robertson, 1990, 1991; 2009 Canadian Geotechnical Journal
    # 9 zones CPTu classification
    soil_type = []
    SBT_Zone = []

    for idx, i in enumerate(Ic_m):
        # Zone 1 (Sensitive Clays and Silts)
        if Qtn[idx] < 12 * np.exp(-1.4 * F[idx]):
            soil_type.append('Sensitive Clays and Silts')
            SBT_Zone.append(1)

        # Zone 9 & Zone 8 (Very Stiff OC)
        elif Qtn[idx] >= (1 / (0.006 * (F[idx] - 0.9) - 0.0004 * (F[idx] - 0.9) ** 2 - 0.002)):
            if F[idx] > 4.5:
                soil_type.append('Very Stiff OC clay to silt')
                SBT_Zone.append(9)
            elif 1.5 < F[idx] <= 4.5:
                soil_type.append('Very Stiff OC clay to clayey sand')
                SBT_Zone.append(8)

        # Zone 7 (Sands with gravels) - 放在 Zone 8/9 外部，確保它獨立
        if i <= 1.31:
            soil_type.append('Sands with gravels')
            SBT_Zone.append(7)
        
        # 其他類別 (根據 Ic_m)
        elif 1.31 < i <= 2.05:
            soil_type.append('Sands: clean to silt')
            SBT_Zone.append(6)

        elif 2.05 < i <= 2.6:
            soil_type.append('Sandy mixtures')
            SBT_Zone.append(5)

        elif 2.6 < i <= 2.95:
            soil_type.append('Silty mixtures')
            SBT_Zone.append(4)

        elif 2.95 < i <= 3.6:
            soil_type.append('Clays')
            SBT_Zone.append(3)

        elif i > 3.6:
            soil_type.append('Organic soils')
            SBT_Zone.append(2)

    return soil_type, SBT_Zone

def est_soil_behavior(Ic_m):
    soil_be = np.full(len(Ic_m), "", dtype=object)  # 預設長度與 `Ic_m` 相同

    mask_drained = Ic_m <= 2.54
    mask_undrained = Ic_m > 2.54

    soil_be[mask_drained] = "Drained"
    soil_be[mask_undrained] = "Undrained"

    return soil_be  # 這樣 `soil_behavior` 的長度一定會等於 `Ic_m`

# Calculate the effective friction angle based on soil behavior
def phi_eff_cal(soil_behavior, Q, Bq, Qtn):
    """
    Compute effective friction angle (φ_eff):
    - If 'Drained' -> Estimated from Qtn
    - If 'Undrained' -> Uses Bq and Q according to NTH solution
    """
    #print("soil_behavior length:", len(soil_behavior))
    #print("Q length:", len(Q))
    #print("Bq length:", len(Bq))
    #print("Qtn length:", len(Qtn))
    phi_eff = np.zeros_like(Qtn, dtype=np.float64)  # Initialize array

    # Drained: φ = 25 * Qtn^0.1
    mask_drained = np.array(soil_behavior) == "Drained"
    phi_eff[mask_drained] = 25 * Qtn[mask_drained] ** 0.1

    # Undrained case
    mask_undrained = np.array(soil_behavior) == "Undrained"
    mask_bq_high = (Bq >= 0.05) & (Bq < 1) & mask_undrained
    mask_bq_low = (Bq < 0.05) & mask_undrained

    # Case 1: 0.05 ≤ Bq < 1
    phi_eff[mask_bq_high] = (
        29.5 * (Bq[mask_bq_high] ** 0.121) *
        (0.256 + 0.336 * Bq[mask_bq_high] + np.log10(Q[mask_bq_high]))
    )

    # Case 2: Bq < 0.05
    phi_eff[mask_bq_low] = 8.18 * np.log(2.13 * Q[mask_bq_low])

    return phi_eff


# Function to estimate Overconsolidation Ratio (OCR)
def OCR_estimation(Ic_m, qnet, sigma_eff):
    """
    Compute OCR based on Ic_m, qnet, and sigma_eff.
    - m_p is computed from Robertson & Wride (1998) equation.
    - Yield stress (σ_yield) estimated using qnet and m_p.
    - OCR = σ_yield / σ_eff
    """
    m_p = 1 - (0.28 / (1 + (Ic_m / 2.65) ** 25))

    yield_sig = 0.33 * qnet**m_p * (101.325 / 100) ** (1 - m_p)
    OCR = yield_sig / sigma_eff  # Compute OCR

    return m_p, yield_sig, OCR
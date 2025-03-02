import numpy as np
import matplotlib.pyplot as plt
class CPTuProcessor:
    def __init__(self, qt, fs, u2, depth, GWT=0):
        """
        Initialize the CPTu Processor with input data.
        qt, fs, u2, depth: numpy arrays of CPTu data
        GWT: Groundwater table depth
        Stress Unit: kPa
        """
        self.qt = np.asarray(qt)
        self.fs = np.asarray(fs)
        self.u2 = np.asarray(u2)
        self.depth = np.asarray(depth)
        self.GWT = GWT

        # Compute all necessary parameters
        self.compute_all()

    def unit_weight_cal(self, sleeve_fric):
        return 12 + 1.5 * np.log(sleeve_fric + 0.1)

    def total_stress_cal(self, unit_weight, depth):
        unit_weight = np.asarray(unit_weight, dtype=np.float64)
        depth = np.asarray(depth, dtype=np.float64)
        layer_thickness = np.diff(depth, prepend=0)
        return np.cumsum(layer_thickness * unit_weight)

    def u0_cal(self, depth, GWT=0):
        return np.where(depth <= GWT, 0, (depth - GWT) * 9.81)

    def sigma_eff_cal(self, sigma_v0, u0):
        return sigma_v0 - u0

    def qnet_deltau(self, qt, sigma_v0, u0, u2):
        return qt - sigma_v0, u2 - u0

    def Normalized_piez_paras(self, q_net, sigma_eff, fs, delta_u):
        Q = q_net / sigma_eff
        F = 100 * fs / q_net
        Bq = np.maximum(delta_u / q_net, 0)
        return Q, F, Bq

    def I_c_calculation(self, Q, F):
        Q = np.where(Q > 0, Q, 1e-6)
        F = np.where(F > 0, F, 1e-6)
        return np.sqrt((3.47 - np.log10(Q)) ** 2 + (1.22 + np.log10(F)) ** 2)

    def Qtn_n(self, I_c, sigma_eff, q_net):
        n = 0.381 * I_c + 0.05 * (sigma_eff / 101.325) - 0.15
        return n, (q_net / 101.325) * (101.325 / sigma_eff) ** n

    def Ic_m_calculation(self, Qtn, F):
        return np.sqrt((3.47 - np.log10(Qtn)) ** 2 + (1.22 + np.log10(F)) ** 2)

    def classification_soil(self, Ic_m, F, Qtn):
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

    def est_soil_behavior(self, Ic_m):
        return np.where(Ic_m <= 2.54, "Drained", "Undrained")

    def phi_eff_cal(self, soil_behavior, Q, Bq, Qtn):
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

    def OCR_estimation(self, Ic_m, qnet, sigma_eff):
        m_p = 1 - (0.28 / (1 + (Ic_m / 2.65) ** 25))
        yield_sig = 0.33 * qnet ** m_p * (101.325 / 100) ** (1 - m_p)
        return m_p, yield_sig, yield_sig / sigma_eff

    def lateral_stress_coefficient(self,phi_eff,OCR):
        return (1-np.sin(np.deg2rad(phi_eff)))*(OCR**(np.sin(np.deg2rad(phi_eff))))

    def compute_all(self):
        self.gamma_t = self.unit_weight_cal(self.fs)
        self.sigma_v0 = self.total_stress_cal(self.gamma_t, self.depth)
        self.u0 = self.u0_cal(self.depth, self.GWT)
        self.sigma_eff = self.sigma_eff_cal(self.sigma_v0, self.u0)
        self.q_net, self.delta_u = self.qnet_deltau(self.qt, self.sigma_v0, self.u0, self.u2)
        self.Q, self.F, self.Bq = self.Normalized_piez_paras(self.q_net, self.sigma_eff, self.fs, self.delta_u)
        self.I_c = self.I_c_calculation(self.Q, self.F)
        self.n, self.Qtn = self.Qtn_n(self.I_c, self.sigma_eff, self.q_net)
        self.Ic_m = self.Ic_m_calculation(self.Qtn, self.F)
        self.soil_type, self.SBT_Zone = self.classification_soil(self.Ic_m, self.F, self.Qtn)
        self.soil_behavior = self.est_soil_behavior(self.Ic_m)
        self.phi_eff = self.phi_eff_cal(self.soil_behavior, self.Q, self.Bq, self.Qtn)
        self.m_p, self.yield_sig, self.OCR = self.OCR_estimation(self.Ic_m, self.q_net, self.sigma_eff)
        self.K0=self.lateral_stress_coefficient(self.phi_eff,self.OCR)

    def plot_cpt_readings(self):
        """Plot qt, fs, and u2 vs depth."""
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3,figsize=[9,6], sharey=True, dpi=288)
        plt.gca().invert_yaxis()
        ax0.plot(self.qt, self.depth, color='r')
        ax0.set_title(r'Cone Resistance, $q_t$', fontsize=8)
        ax0.set(xlabel='kPa', ylabel='Depth, m')
        ax1.plot(self.fs, self.depth, color='k')
        ax1.set_title(r'Sleeve Friction, $f_s$', fontsize=8)
        ax1.set(xlabel='kPa')
        ax2.plot(self.u2, self.depth, color='b')
        ax2.set_title(r'Porewater Pressure, $u_2$', fontsize=8)
        ax2.set(xlabel='kPa')
        for ax in [ax0, ax1, ax2]:
            ax.minorticks_on()
            ax.grid(which='both', alpha=0.3)
        plt.show()
        self.fig_CPT_reading = fig

    def plot_stresses(self):
        """Plot sigma_v0, u0, and sigma_eff vs depth."""
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3,figsize=[9,6], sharey=True, dpi=288)
        plt.gca().invert_yaxis()
        ax0.plot(self.sigma_v0, self.depth, color='k', label='Total stress')
        ax0.set_title(r'Total Stress, $\sigma_{v0}$', fontsize=8)
        ax0.set(xlabel='kPa', ylabel='Depth, m')
        ax1.plot(self.u0, self.depth, color='b', label='Hydrostatic pressure')
        ax1.set_title(r'Hydrostatic pressure, $u_0$', fontsize=8)
        ax1.set(xlabel='kPa')
        ax2.plot(self.sigma_eff, self.depth, color='r', label='Effective stress')
        ax2.set_title(r'Effective Stress, $\sigma_{eff}$', fontsize=8)
        ax2.set(xlabel='kPa')
        for ax in [ax0, ax1, ax2]:
            ax.minorticks_on()
            ax.grid(which='both', alpha=0.3)
        plt.show()
        self.fig_stresses = fig

    def plot_soil_parameters(self):
        """Plot gamma_t, phi_eff, yield_sig, OCR vs depth."""
        fig, (ax0, ax1, ax2, ax3,ax4) = plt.subplots(1, 5,figsize=[15,8], sharey=True, dpi=288)
        plt.gca().invert_yaxis()
        ax0.plot(self.gamma_t, self.depth, color='k')
        ax0.set_title(r'Unit Weight, $\gamma_{t}$', fontsize=8)
        ax0.set(xlabel='kPa', ylabel='Depth, m')
        ax1.plot(self.phi_eff, self.depth, color='b')
        ax1.set_title(r'Friction Angle, $\phi$', fontsize=8)
        ax1.set(xlabel=r'$(\cdot)^{\degree}$')
        ax2.plot(self.yield_sig, self.depth, color='r')
        ax2.set_title(r"Yield Stress, $\sigma_{p}'$", fontsize=8)
        ax2.set(xlabel='kPa')
        ax3.plot(self.OCR, self.depth, color='g')
        ax3.set_title(r'Overconsolidation Ratio', fontsize=8)
        ax3.set(xlabel='')
        ax4.plot(self.K0, self.depth, color='violet')
        ax4.set_title(r'Lateral Stress Coefficient, $K_0$', fontsize=8)
        ax4.set(xlabel='')
        for ax in [ax0, ax1, ax2, ax3,ax4]:
            ax.minorticks_on()
            ax.grid(which='both', alpha=0.3)
        plt.show()
        self.fig_soil_parameters = fig
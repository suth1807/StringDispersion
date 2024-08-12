
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate
import scipy.integrate as integrate
from scipy.optimize import newton
from scipy.optimize import bisect
import pathlib

import wave

from oct2py import octave
import scipy.signal as signal

class Harmonica:
    def __init__(self):
        pass


class Reed:
    def __init__(self, reed_blown_closed_: bool, m_0: float = 6.154e-6, R: float = 180.737e-6, k_0: float = 47.9, L_r: float = 12.95e-3):
        self.deb_l = []
        self.__reset()
        # if true then (-,+) reed, else (+,-). Fletcher's conventions
        self.reed_blown_closed = reed_blown_closed_
        self.load_harmonica_reed_params(m_0, R, k_0, L_r)

    def __reset(self):
        self.R = 0.0
        self.r = 0.0
        self.fs = 44100.0
        self.m_0 = 1.0  # 1.5e-4
        self.k_0 = 1000.0  # 1140.0


    def compute_useful_section(self):
        delta_mm = 0.001
        delta_m = 0.001e-3
        hn_range = np.arange(-5.0, 5.0, delta_mm) * 1.0e-3
        self.su_computed = np.zeros_like(hn_range)

        Lr_m = self.L_r

        def get_eff_hn(h_n):
            if h_n > self.e_t:
                h_eff_s = h_n - self.e_t
            elif h_n < -self.e_s:
                h_eff_s = -(h_n + self.e_s)
            else:
                h_eff_s = 0.0
            return h_eff_s

        def side_len_givenh_n(h_n):
            def side_len(s: float):
                psi = math.pow(s / Lr_m, self.reed_shape_power)
                hn_s = (h_n - self.h_n000) * psi + self.h_n000
                h_eff_s = get_eff_hn(hn_s)
                return math.sqrt(h_eff_s * h_eff_s + self.h_min * self.h_min)
            return side_len

        for i, h_n in enumerate(hn_range):
            integr = integrate.quad(side_len_givenh_n(h_n), 0, Lr_m)
            term2 = 2.0 * integr[0]
            h_eff = h_n  # get_eff_hn(h_n)
            term1 = math.sqrt(h_eff * h_eff + self.h_min *
                              self.h_min) * self.W_r
            self.su_computed[i] = (term2 + term1)
        # plt.plot(hn_range, self.su_computed)
        # plt.show()
        self.su_func = interpolate.interp1d(
            hn_range, self.su_computed, 'linear')

        self.su_grad = np.gradient(self.su_computed, delta_m)
        self.su_grad_func = interpolate.interp1d(
            hn_range, self.su_grad, 'linear')

    def load_harmonica_reed_params(self, m_0: float, R: float, k_0: float, L_r: float):
        # See A Proposal for a Minimal model of Free Reeds, Laurent Millot, Acta Acustica, Jan 2007
        self.L_r = L_r
        # pump flow
        self.W_r = 2.1e-3
        self.reed_shape_power = 1.5
        self.S_r = self.W_r * self.L_r / (self.reed_shape_power + 1)

        self.R = R
        self.m_0 = m_0
        self.k_0 = k_0
        self.g_r = self.R / self.m_0
        self.w_0_sq = self.k_0 / self.m_0

        print(
            f'{math.sqrt(self.k_0 / self.m_0) / (2 * math.pi )} Hz is the eigen frequency')

        # Rest departure from flat reed position
        self.delta_h_n00 = 528.0e-6
        # Reed Thickness
        self.e_t = 110.0e-6
        # Support thickness
        self.e_s = 900.0e-6
        # clearance gap
        self.h_min = 50.0e-6
        if self.reed_blown_closed:
            self.h_n00 = -self.delta_h_n00 - self.e_s
            self.h_n000 = -self.e_s
        else:
            self.h_n00 = self.delta_h_n00 + self.e_t
            self.h_n000 = self.e_t

        self.compute_useful_section()

        self.a_0 = 0.5 / (self.fs * self.S_r)
        self.a_1 = -self.w_0_sq * self.S_r * 0.5 / self.fs
        self.a_2 = -self.g_r * 0.5 / self.fs
        self.mu_r = self.m_0 / self.S_r
        self.a_3 = self.S_r * 0.5 / (self.mu_r * self.fs)

    def test_step(self, U_prev: float, y_prev: float, p_2: float, p_2_prev: float):
        U = (self.a_0 * self.a_1 + self.a_2 + 1) * \
            U_prev / (1 - self.a_0 * self.a_1 - self.a_2) + (2 * self.a_1 * y_prev) / (1 - self.a_0 * self.a_1 - self.a_2) + \
            self.a_3 * (p_2 + p_2_prev) / (1 - self.a_0 * self.a_1 - self.a_2)
        y = self.a_0 * (U + U_prev) + y_prev
        return (y, U, y + self.h_n00)

    def test_step_derivative(self, U_prev: float, y_prev: float, p_2: float, p_2_prev: float):
        du_by_dp2 = self.a_3 / (1 - self.a_0 * self.a_1 - self.a_2)
        dy_by_dp2 = self.a_0 * du_by_dp2
        return (du_by_dp2, dy_by_dp2)

    def test(self):
        U = 0.0
        y = 0.0
        p_2_prev = 0.0
        p_2 = 1.0
        osc = np.zeros(10000)
        for i in range(10000):
            (y, U, _) = self.test_step(U, y, p_2, p_2_prev)
            p_2_prev = p_2
            osc[i] = y

        plt.plot(osc)
        plt.show()


class CoupledReed:
    class State:
        def __init__(self):
            self.__reset__()

        def __reset__(self):
            self.p_2 = 0
            self.p_1 = 0
            self.U_r = 0
            self.U = 0
            self.y = 0
            self.h_n = 0
            self.u_0 = 0
            self.f = 0

        def update(self, other):
            self.p_2 = other.p_2
            self.p_1 = other.p_1
            self.U_r = other.U_r
            self.U = other.U
            self.y = other.y
            self.h_n = other.h_n
            self.u_0 = other.u_0
            self.f = other.f

    def __init__(self, reed: Reed, S_2):
        self.reed = reed
        self.rho = 1.293  # Air density
        self.c_0 = 343.0  # Speed of sound
        self.S_0 = 30.0e-6
        self.S_1 = 800.0e-6
        self.L_2 = 20e-3
        self.S_2 = S_2  # 25e-6
        if(reed.reed_blown_closed):
            self.L_1 = 8e-2  # 8CM
        else:
            self.L_1 = 1.5e-2
        self.V_1 = self.L_1 * self.S_1
        self.fs = reed.fs

        self.a_0 = self.V_1 / (self.c_0 * self.c_0)
        self.a_1 = self.rho * self.L_2 / self.S_2
        self.alpha = 0.611  # Vena Contracta
        self.a_2 = math.sqrt(2.0 / self.rho) * self.alpha

        self.current_state = CoupledReed.State()
        self.prev_state = CoupledReed.State()
        #val = self.a_0 * self.L_2 * reed.w_0_sq / self.S_2
        pass

    def evaluate_NL(self, p_2: float):
        self.current_state.p_2 = p_2
        (self.current_state.y, self.current_state.U_r, self.current_state.h_n) = self.reed.test_step(
            self.prev_state.U_r, self.prev_state.y,  self.current_state.p_2, self.prev_state.p_2)
        p_2_abs = abs(self.current_state.p_2)
        if self.current_state.p_2 < 0.0:
            self.current_state.U = self.current_state.U_r - self.a_2 * \
                self.reed.su_func(self.current_state.h_n) * math.sqrt(p_2_abs)
        else:
            self.current_state.U = self.current_state.U_r + self.a_2 * \
                self.reed.su_func(self.current_state.h_n) * math.sqrt(p_2_abs)

        self.current_state.p_1 = self.current_state.p_2 + self.a_1 * \
            self.fs * (self.current_state.U - self.prev_state.U)

        f = self.a_0 * self.fs * (self.current_state.p_1 - self.prev_state.p_1) - \
            self.rho * (self.current_state.u_0 - self.current_state.U)
        self.current_state.f = f
        return f

    def evaluate_NL_der(self, p_2: float):
        self.current_state.p_2 = p_2
        du_r_dp2, dy_dp2 = self.reed.test_step_derivative(
            self.prev_state.U_r, self.prev_state.y,  self.current_state.p_2, self.prev_state.p_2)
        p_2_abs = abs(self.current_state.p_2)
        # prevent division by zero
        p_2_abs = max(1.0e-8, p_2_abs)
        if p_2 < 0.0:
            du_dp2 = du_r_dp2 - self.a_2 * self.reed.su_grad_func(self.current_state.h_n) * dy_dp2 * math.sqrt(
                p_2_abs) + 0.5 * self.a_2 * self.reed.su_func(self.current_state.h_n) / math.sqrt(p_2_abs)
        else:
            du_dp2 = du_r_dp2 + self.a_2 * self.reed.su_grad_func(self.current_state.h_n) * dy_dp2 * math.sqrt(
                p_2_abs) + 0.5 * self.a_2 * self.reed.su_func(self.current_state.h_n) / math.sqrt(p_2_abs)
        dp1_by_dp2 = 1 + self.a_1 * self.fs * du_dp2

        df = self.a_0 * self.fs * dp1_by_dp2 + self.rho * du_dp2
        return df

    def step(self, v_0: float):
        self.current_state.u_0 = v_0 * self.S_0

        (root, r) = newton(self.evaluate_NL, self.prev_state.p_2,
                           fprime=self.evaluate_NL_der, full_output=True, disp=False)
        if r.converged == False:
            (root, r) = newton(self.evaluate_NL, self.prev_state.p_2-100,
                               x1=self.prev_state.p_2+100, full_output=True, disp=False)

        self.prev_state.update(self.current_state)
        # print(f'{r.iterations}')
        return root

    def test():
        scale = 1.0
        Rscale = 1.0
        L_r_scale = 1.0
        S_2_scale = 1.0
        m_0: float = 6.154e-6 * scale
        R: float = 180.737e-6 * Rscale
        k_0: float = 47.9 / scale
        L_r: float = 12.95e-3 * L_r_scale
        S_2 = 25e-6 * S_2_scale

        s: Reed = Reed(True, m_0, R, k_0, L_r)
        coupled_reed: CoupledReed = CoupledReed(s, S_2)
        num_steps = 44100

        oscillations = np.zeros(num_steps)

        vel = 3.5
        velocity = np.array([vel, vel, vel,   vel,     vel,     0.0,  0.0])
        time = np.array([0.0, 100, 440.0, 1000.0,  38000.0, 40400, num_steps])
        f_int = interpolate.interp1d(time, velocity, 'linear')
        full_time_points = np.arange(0.0, num_steps)
        full_v_vals = f_int(full_time_points)
        ns = np.random.normal(0.0, 1.0, num_steps)
        ns_level = 0.02
        for i in range(num_steps):
            v = full_v_vals[i]*(1.0 + ns_level * ns[i])
            oscillations[i] = coupled_reed.step(v)

        plt.plot(oscillations)
        plt.show()

        min = np.min(oscillations)
        max = np.max(oscillations)
        target_min = -32768
        target_max = 32767
        oscillations = (oscillations - min) * (target_max -
                                               target_min) / (max - min) + target_min
        oscillations = oscillations * 0.5
        oscillations = oscillations - oscillations[0]
        oscillations = np.round(oscillations)
        oscillations = oscillations.astype(np.int16)

        oscillations = oscillations.tobytes()
        padd_begin = np.zeros(4000).astype(np.int16).tobytes()
        padd_end = np.zeros(4000).astype(np.int16).tobytes()
        with wave.open("output.wav", mode="wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(44100)
            wav_file.writeframes(padd_begin)
            wav_file.writeframes(oscillations)
            wav_file.writeframes(padd_end)


def generate_harmonica(debugMode=False):
    CoupledReed.test()


if __name__ == '__main__':
    generate_harmonica(True)

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.size"] = 14
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10


def compute_initial_estimate(xe: np.ndarray, Ax: np.ndarray, mu: float):
    batch_size = xe.shape[0]
    x0_bar = np.zeros((batch_size, 4))
    x0_bar[:, 0] = xe[:, 0] - Ax[:, 0]
    mu_bar = compute_mu_bar(x0_bar, mu)
    """問題2-1: リアプノフ軌道の初期値の推定
    リアプノフ軌道の初期値のy方向の速度vyの計算を実装しなさい．
    """
    nu = np.sqrt( -(mu_bar-2-np.sqrt(9*mu_bar**2 - 8*mu_bar))/2 )
    tau = -(nu**2 + 2*mu_bar +1)/(2*nu)
    x0_bar[:,3] = - Ax[:,0] * nu[:,0] * tau[:,0]
    return x0_bar


def compute_delta_vy0(dx_dt, Phi):
    batch_size = dx_dt.shape[0]
    delta_vy0 = np.zeros((batch_size, 4), dtype=np.float64)
    """問題2-2: Differential Correction
    補正値δvy0の計算を実装しなさい．
    """
    delta_vy0[:,3] = dx_dt[:,0] * (Phi[:,2,3] - dx_dt[:,2] / (dx_dt[:,1] + 1e-11) * Phi[:,1,3]) ** (-1)
    return delta_vy0

def compute_energy(x: np.ndarray, mu: float):
    E = np.zeros((x.shape[0], 1), dtype=np.float64)
    U_bar = compute_U_bar(x, mu)
    E[:, 0] = (x[:, 2] ** 2 + x[:, 3] ** 2) / 2 + U_bar[:, 0]
    return E


def compute_mu_bar(x: np.ndarray, mu: float):
    mu1 = 1 - mu
    mu2 = mu

    r1 = np.zeros((x.shape[0], 1), dtype=np.float64)
    r2 = np.zeros((x.shape[0], 1), dtype=np.float64)

    r1[:, 0] = np.sqrt((x[:, 0] + mu2) ** 2 + x[:, 1] ** 2)
    r2[:, 0] = np.sqrt((x[:, 0] - mu1) ** 2 + x[:, 1] ** 2)

    mu_bar = mu1 / r1 ** 3 + mu2 / r2 ** 3
    return mu_bar


def compute_U(x: np.ndarray, mu: float):
    mu1 = 1 - mu
    mu2 = mu

    U = np.zeros((x.shape[0], 1), dtype=np.float64)
    r1 = np.zeros((x.shape[0], 1), dtype=np.float64)
    r2 = np.zeros((x.shape[0], 1), dtype=np.float64)

    r1[:, 0] = np.sqrt((x[:, 0] + mu2) ** 2 + x[:, 1] ** 2)
    r2[:, 0] = np.sqrt((x[:, 0] - mu1) ** 2 + x[:, 1] ** 2)
    U[:, 0] = -mu1 / r1[:, 0] - mu2 / r2[:, 0] - mu1 * mu2 / 2
    return U


def compute_U_bar(x: np.ndarray, mu: float):
    U_bar = np.zeros((x.shape[0], 1), dtype=np.float64)
    U = compute_U(x, mu)
    U_bar[:, 0] = -(x[:, 0] ** 2 + x[:, 1] ** 2) / 2 + U[:, 0]
    return U_bar


def compute_dU(x: np.ndarray, mu: float):
    mu1 = 1 - mu
    mu2 = mu

    r1 = np.zeros((x.shape[0], 1), dtype=np.float64)
    r2 = np.zeros((x.shape[0], 1), dtype=np.float64)

    dr1 = x[:, :2].copy()
    dr1[:, 0] = dr1[:, 0] + mu2
    r1[:, 0] = np.sqrt(np.sum(dr1 ** 2, axis=1))
    dr1 = dr1 / r1

    dr2 = x[:, :2].copy()
    dr2[:, 0] = dr2[:, 0] - mu1
    r2[:, 0] = np.sqrt(np.sum(dr2 ** 2, axis=1))
    dr2 = dr2 / r2

    dU = mu1 * dr1 / r1 ** 2 + mu2 * dr2 / r2 ** 2
    return dU


def compute_dU_bar(x: np.ndarray, mu: float):
    dU = compute_dU(x, mu)
    dU_bar = np.zeros((x.shape[0], 2), dtype=np.float64)
    dU_bar[:, 0] = dU[:, 0] - x[:, 0]
    dU_bar[:, 1] = dU[:, 1] - x[:, 1]
    return dU_bar


def compute_ddU_bar(x: np.ndarray, mu: float):
    mu1 = 1 - mu
    mu2 = mu

    r1 = np.zeros((x.shape[0], 1), dtype=np.float64)
    r2 = np.zeros((x.shape[0], 1), dtype=np.float64)

    mu_bar = compute_mu_bar(x, mu)
    r1[:, 0] = np.sqrt((x[:, 0] + mu2) ** 2 + x[:, 1] ** 2)
    r2[:, 0] = np.sqrt((x[:, 0] - mu1) ** 2 + x[:, 1] ** 2)

    mu_bar_ = np.zeros_like(mu_bar, dtype=np.float64)
    mu_bar_[:, 0] = (
        mu1 * (x[:, 0] + mu2) ** 2 / r1[:, 0] ** 5
        + mu2 * (x[:, 0] - mu1) ** 2 / r2[:, 0] ** 5
    )

    ddU_bar = np.zeros((x.shape[0], 2, 2), dtype=np.float64)
    ddU_bar[:, 0, 0] = -1 + mu_bar[:, 0] - 3 * mu_bar_[:, 0]
    ddU_bar[:, 1, 1] = (
        -1
        + mu_bar[:, 0]
        - 3 * x[:, 1] ** 2 * (mu1 / r1[:, 0] ** 5 + mu2 / r2[:, 0] ** 5)
    )
    ddU_bar[:, 0, 1] = ddU_bar[:, 1, 0] = -3 * (
        (x[:, 0] + mu2) * x[:, 1] * mu1 / r1[:, 0] ** 5
        + (x[:, 0] - mu1) * x[:, 1] * mu2 / r2[:, 0] ** 5
    )
    return ddU_bar


def inner_product(A: np.ndarray, B: np.ndarray):
    C = np.zeros((A.shape[0], A.shape[1], B.shape[2]), dtype=np.float64)
    for idx_i in range(A.shape[0]):
        C[idx_i] = np.dot(A[idx_i], B[idx_i])
    return C


def ode_pcr3bp(x: np.ndarray, mu: float):
    dU_bar = compute_dU_bar(x, mu)
    dx_dt = np.zeros_like(x, dtype=np.float64)
    dx_dt[:, 0] = x[:, 2]
    dx_dt[:, 1] = x[:, 3]
    dx_dt[:, 2] = 2 * x[:, 3] - dU_bar[:, 0]
    dx_dt[:, 3] = -2 * x[:, 2] - dU_bar[:, 1]
    return dx_dt


def linear_ode_coef_matrix(x: np.ndarray, mu: float):
    ddU_bar = compute_ddU_bar(x, mu)
    A = np.zeros((x.shape[0], 4, 4), dtype=np.float64)
    A[:, 0, 2] = 1
    A[:, 1, 3] = 1
    A[:, 2, 0] = -ddU_bar[:, 0, 0]
    A[:, 2, 1] = -ddU_bar[:, 0, 1]
    A[:, 2, 3] = 2
    A[:, 3, 0] = -ddU_bar[:, 1, 0]
    A[:, 3, 1] = -ddU_bar[:, 1, 1]
    A[:, 3, 2] = -2
    return A


def linear_ode_pcr3bp(x: np.ndarray, Phi: np.ndarray, mu: float):
    A = linear_ode_coef_matrix(x, mu)
    return inner_product(A, Phi)


def runge_kutta(t: np.ndarray, x: np.ndarray, dt: np.ndarray, mu: float):
    """Runge-Kutta method (https://ja.wikipedia.org/wiki/ルンゲ＝クッタ法)
    Args:
        t (np.ndarray): (n-batch, m-dim) time t
        x (np.ndarray): (n-batch, m-dim) vector at time t
        dt (np.ndarray): (n-batch, m-dim) time interval
        mu (float): mass
    Returns:
        np.ndarray: time t + dt
        np.ndarray: n-dim vector at time x + dx
    """
    k1 = ode_pcr3bp(x, mu)
    k2 = ode_pcr3bp(x + dt / 2 * k1, mu)
    k3 = ode_pcr3bp(x + dt / 2 * k2, mu)
    k4 = ode_pcr3bp(x + dt * k3, mu)
    return t + dt, x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def runge_kutta_with_transition_matrix(
    t: np.ndarray,
    x: np.ndarray,
    Phi: np.ndarray,
    dt: np.ndarray,
    mu: float,
):
    """Runge-Kutta method (https://ja.wikipedia.org/wiki/ルンゲ＝クッタ法)
    Args:
        t (np.ndarray): (n-batch, 1) time t
        x (np.ndarray): (n-batch, m-dim) vector at time t
        Phi (np.ndarray): (n-batch, m-dim, m-dim) State transition matrix
        dt (np.ndarray): (n-batch, m-dim) time interval
        mu (float): mass
    Returns:
        np.ndarray: (n-batch, 1) t + dt
        np.ndarray: (n-batch, m-dim) x + dx
        np.ndarray: (n-dim, 1) Phi + dPhi
    """
    k1_x = ode_pcr3bp(x, mu)
    k2_x = ode_pcr3bp(x + dt / 2 * k1_x, mu)
    k3_x = ode_pcr3bp(x + dt / 2 * k2_x, mu)
    k4_x = ode_pcr3bp(x + dt * k3_x, mu)

    dt_ = np.ones_like(Phi, dtype=np.float64)
    for idx in range(Phi.shape[0]):
        dt_[idx] = dt[idx, 0]
    k1_Phi = linear_ode_pcr3bp(x, Phi, mu)
    k2_Phi = linear_ode_pcr3bp(x + dt / 2 * k1_x, Phi + dt_ / 2 * k1_Phi, mu)
    k3_Phi = linear_ode_pcr3bp(x + dt / 2 * k2_x, Phi + dt_ / 2 * k2_Phi, mu)
    k4_Phi = linear_ode_pcr3bp(x + dt * k3_x, Phi + dt_ * k3_Phi, mu)

    x_next = x + dt / 6 * (k1_x + 2 * k2_x + 2 * k3_x + k4_x)
    Phi_next = Phi + dt_ / 6 * (k1_Phi + 2 * k2_Phi + 2 * k3_Phi + k4_Phi)

    return t + dt, x_next, Phi_next


def differential_correction(
    x0_bar: np.ndarray,
    dt: np.ndarray,
    mu: np.ndarray,
    d: np.ndarray,
    epsilon: np.ndarray,
    max_iterations: int = 1000,
):
    # 初期化
    xn_bar = x0_bar.copy()
    global_valid = np.ones(x0_bar.shape[0], dtype=np.int64)
    for n_iteration in range(max_iterations):
        # 微分補正を実行するバッチの抽出
        x = xn_bar[global_valid == 1].copy()
        t = np.zeros((x.shape[0], 1), dtype=np.float64)
        dt_ = dt[global_valid == 1].copy()
        d_ = d[global_valid == 1].copy()
        epsilon_ = epsilon[global_valid == 1].copy()

        # 初期化
        Phi = np.zeros((x.shape[0], x.shape[1], x.shape[1]), dtype=np.float64)
        for idx_i in range(Phi.shape[0]):
            for idx_j in range(Phi.shape[1]):
                Phi[idx_i, idx_j, idx_j] = 1.0
        valid = np.ones(x.shape[0], dtype=np.int64)

        # Lyapunov軌道を半周期分だけ数値計算
        while np.sum(valid) >= 1:
            # 数値計算を実行するバッチの抽出
            t_valid = t[valid == 1].copy()
            x_valid = x[valid == 1].copy()
            Phi_valid = Phi[valid == 1].copy()
            dt_valid = dt_[valid == 1].copy()
            d_valid = d_[valid == 1].copy()
            epsilon_valid = epsilon_[valid == 1].copy()

            # 次の時刻tにおける相空間x, 状態遷移行列Phiを計算
            t_next, x_next, Phi_next = runge_kutta_with_transition_matrix(
                t=t_valid,
                x=x_valid,
                Phi=Phi_valid,
                dt=dt_valid,
                mu=mu,
            )

            # 終了判定doneと再計算判定retryの計算
            done = np.zeros(x_next.shape[0], dtype=np.int64)
            retry = np.zeros(x_next.shape[0], dtype=np.int64)
            done[x_valid[:, 1] > 0] = 1
            done[x_next[:, 1] > 0] = 0
            retry[x_valid[:, 1] > 0] = 1
            retry[x_next[:, 1] > 0] = 0
            done[np.abs(x_valid[:, 1]) >= epsilon_valid[:, 0]] = 0
            retry[np.abs(x_valid[:, 1]) < epsilon_valid[:, 0]] = 0

            # パラメータの更新
            dt_next = dt_valid
            dt_next[retry == 1] = 0.99 * dt_next[retry == 1].copy()
            dt_[valid == 1] = dt_next
            t_next[retry == 1] = t_valid[retry == 1].copy()
            t[valid == 1] = t_next.copy()
            x_next[retry == 1] = x_valid[retry == 1].copy()
            x[valid == 1] = x_next.copy()
            Phi_next[retry == 1] = Phi_valid[retry == 1].copy()
            Phi[valid == 1] = Phi_next.copy()

            # 有効なバッチの判定validの更新
            valid[valid == 1] = 1 - done

        # 有効なバッチの判定global_valid
        global_valid_valid = global_valid[global_valid == 1].copy()
        global_valid_valid[np.abs(x[:, 2]) < d_valid[:, 0]] = 0
        global_valid[global_valid == 1] = global_valid_valid
        if np.sum(global_valid) == 0:
            break

        # 微分補正(differential correction)の計算
        t_valid = t[global_valid_valid == 1].copy()
        x_valid = x[global_valid_valid == 1].copy()
        Phi_valid = Phi[global_valid_valid == 1].copy()
        dx_dt_valid = ode_pcr3bp(x_valid, mu)
        delta_vy0 = compute_delta_vy0(dx_dt_valid, Phi_valid)

        if (n_iteration + 1) % 50 == 0:
            scale = 0.5
        else:
            scale = 1.0
        xn_bar[global_valid == 1] = (
            xn_bar[global_valid == 1] - scale * delta_vy0
        )
    return xn_bar


def continuation(
    x0_bar1: np.ndarray,
    x0_bar2: np.ndarray,
    E_target: np.ndarray,
    dt: np.ndarray,
    mu: np.ndarray,
    d: np.ndarray,
    epsilon: np.ndarray,
    converge: np.ndarray,
    max_iterations: int = 1000,
    max_episodes: int = 1000,
):
    n = x0_bar1.shape[0]
    Delta = x0_bar2 - x0_bar1
    x0_bar = x0_bar2 + Delta
    valid = np.ones(n, dtype=np.int64)
    for episode in range(max_episodes):
        # 有効なバッチの取得
        x0_bar1_valid = x0_bar1[valid == 1].copy()
        x0_bar2_valid = x0_bar2[valid == 1].copy()
        Delta_valid = Delta[valid == 1]

        x0_bar_valid = x0_bar[valid == 1].copy()
        dt_valid = dt[valid == 1].copy()
        d_valid = d[valid == 1].copy()
        epsilon_valid = epsilon[valid == 1].copy()
        converge_valid = converge[valid == 1].copy()
        x0_bar_valid = differential_correction(
            x0_bar=x0_bar_valid,
            dt=dt_valid,
            mu=mu,
            d=d_valid,
            epsilon=epsilon_valid,
            max_iterations=max_iterations,
        )

        # エネルギーの計算
        E_valid = compute_energy(x0_bar_valid, mu)
        print(
            f"episode: {episode + 1}, Energy (L1, L2): {E_valid.reshape(-1)}"
        )

        # 終了判定doneと再計算判定retryの計算
        n_valid = x0_bar_valid.shape[0]
        done = np.zeros(n_valid, dtype=np.int64)
        retry = np.zeros(n_valid, dtype=np.int64)
        retry[E_valid[:, 0] > E_target[valid == 1, 0]] = 1
        done[
            (
                np.abs(E_valid[:, 0] - E_target[valid == 1, 0])
                < converge_valid[:, 0]
            )
        ] = 1
        retry[
            (
                np.abs(E_valid[:, 0] - E_target[valid == 1, 0])
                < converge_valid[:, 0]
            )
        ] = 0

        # 有効なバッチの判定validの更新
        valid[valid == 1] = 1 - done
        if np.sum(valid) == 0:
            break

        # パラメータの更新
        x0_bar1_valid[retry == 0] = x0_bar2_valid[retry == 0].copy()
        x0_bar2_valid[retry == 0] = x0_bar_valid[retry == 0].copy()
        Delta_valid[retry == 0] = (
            x0_bar2_valid[retry == 0] - x0_bar1_valid[retry == 0]
        )
        Delta_valid[retry == 1] = 0.5 * Delta_valid[retry == 1]
        x0_bar_valid = x0_bar2_valid + Delta_valid
        x0_bar1[valid == 1] = x0_bar1_valid[done == 0].copy()
        x0_bar2[valid == 1] = x0_bar2_valid[done == 0].copy()
        Delta[valid == 1] = Delta_valid[done == 0].copy()
        x0_bar[valid == 1] = x0_bar_valid[done == 0].copy()
    return x0_bar


def lyapunov_orbit(
    t_start: np.ndarray,
    t_end: np.ndarray,
    x0: np.ndarray,
    dt: np.ndarray,
    mu: float,
):
    # 初期値
    t = t_start.copy()
    x = x0.copy()

    # 実行結果を保存するリスト
    ts = [t.copy()]
    xs = [x.copy()]

    # 数値計算を実行
    valid = np.ones(x0.shape[0], dtype=np.int64)
    while np.sum(valid) > 0:
        # 次の時刻の数値計算を実行
        t_valid = t[valid == 1].copy()
        x_valid = x[valid == 1].copy()
        dt_valid = dt[valid == 1].copy()
        t_next, x_next = runge_kutta(t_valid, x_valid, dt_valid, mu=mu)

        # 終了条件
        done = np.zeros(x_next.shape[0], dtype=np.int64)
        retry = np.zeros(x_next.shape[0], dtype=np.int64)
        done[x_valid[:, 1] < 0] = 1
        done[x_next[:, 1] < 0] = 0
        retry[x_valid[:, 1] < 0] = 1
        retry[x_next[:, 1] < 0] = 0
        done[np.abs(x_valid[:, 1]) >= 1e-11] = 0
        retry[np.abs(x_valid[:, 1]) < 1e-11] = 0

        # tとxの更新
        dt_next = dt_valid
        dt_next[retry == 1] = 0.99 * dt_next[retry == 1].copy()
        dt_valid = dt_next.copy()
        t_next[retry == 1] = t_valid[retry == 1].copy()
        x_next[retry == 1] = x_valid[retry == 1].copy()

        t_end_valid = t_end[valid == 1].copy()
        done[np.abs(t_next[:, 0]) >= np.abs(t_end_valid[:, 0])] = 1

        # tとxの更新
        dt[valid == 1] = dt_valid.copy()
        t[valid == 1] = t_next.copy()
        x[valid == 1] = x_next.copy()

        # 有効判定validの更新
        valid[valid == 1] = 1 - done

        # 次の時刻のtとxをリストに保存
        ts.append(t.copy())
        xs.append(x.copy())

    # リストをnumpy配列に変換
    ts = np.array(ts, dtype=np.float64)
    xs = np.array(xs, dtype=np.float64)

    ts = ts.reshape(*ts.shape[:2]).T
    xs = np.swapaxes(xs, 0, 1)
    return ts, xs


def draw_a_point(ax, x, y, text=None, **kwargs):
    ax.scatter(x, y, **kwargs)
    if text is not None:
        ax.text(x, y + 0.002, text)
    return ax


def draw_laypunov_orbit(ax, xs):
    for idx in range(xs.shape[0]):
        ax.plot(xs[idx, :, 0], xs[idx, :, 1])
    return ax


def main():
    parser = argparse.ArgumentParser(description="Solve Lyapunov Orbit")
    parser.add_argument(
        "--system",
        choices=["sun_jupyter", "earth_moon"],
        required=True,
    )
    parser.add_argument("--Ax1", default=1e-4, type=float)
    parser.add_argument("--Ax2", default=5e-4, type=float)
    parser.add_argument("--dt", default=1e-3, type=float)
    parser.add_argument("-d", default=1e-8, type=float)
    parser.add_argument("--epsilon", default=1e-11, type=float)
    parser.add_argument("--max_iterations", default=1000, type=int)
    parser.add_argument("--max_episodes", default=1000, type=int)
    parser.add_argument("--E_target", default=-1.515, type=float)
    parser.add_argument("--converge", default=1e-7, type=float)
    parser.add_argument("--saveflag", action="store_true")
    parser.add_argument("--savedir", default="")
 

    args = parser.parse_args()

    # PCR3BP
    # パラメータの設定
    if args.system == "sun_jupyter":
        mu = 9.537e-4
        xe = np.array(
            [[0.93236975, 0.0, 0.0, 0.0], [1.06882633, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    elif args.system == "earth_moon":
        mu = 0.01215
        xe = np.array(
            [[0.83691801, 0.0, 0.0, 0.0], [1.15567992, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
    Ax1 = np.array([[args.Ax1], [args.Ax1]], dtype=np.float64)
    Ax2 = np.array([[args.Ax2], [args.Ax2]], dtype=np.float64)
    dt = np.array([[args.dt], [args.dt]], dtype=np.float64)
    d = np.array([[args.d], [args.d]], dtype=np.float64)
    epsilon = np.array([[args.epsilon], [args.epsilon]], dtype=np.float64)
    max_iterations = args.max_iterations
    max_episodes = args.max_episodes
    E_target = np.array([[args.E_target], [args.E_target]], dtype=np.float64)
    converge = np.array([[args.converge], [args.converge]], dtype=np.float64)

    # 半径が小さいリアプノフ軌道の初期値の生成
    xe_ = np.concatenate([xe, xe], axis=0)
    Ax_ = np.concatenate([Ax1, Ax2], axis=0)
    dt_ = np.concatenate([dt, dt], axis=0)
    d_ = np.concatenate([d, d], axis=0)
    epsilon_ = np.concatenate([epsilon, epsilon], axis=0)
    x0_bar = compute_initial_estimate(xe=xe_, Ax=Ax_, mu=mu)
    x0_bar = differential_correction(
        x0_bar=x0_bar,
        dt=dt_,
        mu=mu,
        d=d_,
        epsilon=epsilon_,
        max_iterations=max_iterations,
    )
    x0_bar1, x0_bar2 = np.split(x0_bar, 2)

    # 目的エネルギーでのリアプノフ軌道の初期値の生成
    x0_bar = continuation(
        x0_bar1=x0_bar1,
        x0_bar2=x0_bar2,
        E_target=E_target,
        dt=dt,
        mu=mu,
        d=d,
        epsilon=epsilon,
        max_iterations=max_iterations,
        max_episodes=max_episodes,
        converge=converge,
    )

    # リアプノフ軌道の初期値の出力
    energy = compute_energy(x0_bar, mu)
    for x0_bar_idx, energy_idx in zip(x0_bar, energy):
        print(f"x0_bar: {x0_bar_idx}, energy: {energy_idx}")

    # リアプノフ軌道の計算
    t_start = np.zeros((x0_bar.shape[0], 1), dtype=np.float64)
    t_end = 100 * np.ones((x0_bar.shape[0], 1), dtype=np.float64)
    ts, xs = lyapunov_orbit(
        t_start=t_start, t_end=t_end, x0=x0_bar, dt=dt, mu=mu
    )

    # リアプノフ軌道上のエネルギーの計算
    xs_ = np.concatenate(xs, axis=0)
    energies = compute_energy(xs_, mu)
    ts_L1, ts_L2 = np.split(ts, 2)
    xs_L1, xs_L2 = np.split(xs, 2)
    energies_L1, energies_L2 = np.split(energies, 2)

    # グラフの描画
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(224)

    # Lyapunov軌道
    ax1 = draw_laypunov_orbit(ax1, xs)

    # ラグランジュ点L1, L2
    kwargs = dict(color="black", marker="+", s=50)
    for idx, xe_idx in enumerate(xe):
        text = f"$L_{idx + 1}$"
        ax1 = draw_a_point(
            ax=ax1,
            x=xe_idx[0],
            y=xe_idx[1],
            text=text,
            **kwargs,
        )
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$y$")

    # L1 Lyapunov軌道のエネルギー
    max_idx = np.argmin(np.linalg.norm(xs_L1 - xs_L1[:, [-1]], axis=-1))
    ts_L1 = ts_L1[0, : max_idx + 1]
    energies_L1 = energies_L1[: max_idx + 1]
    ax2.plot(ts_L1, energies_L1, color="black")
    ax2.set_xlim(0, np.max(ts_L1))

    # L2 Lyapunov軌道のエネルギー
    max_idx = np.argmin(np.linalg.norm(xs_L2 - xs_L2[:, [-1]], axis=-1))
    ts_L2 = ts_L2[0, : max_idx + 1]
    energies_L2 = energies_L2[: max_idx + 1]
    ax3.plot(ts_L2, energies_L2, color="black")
    ax3.set_xlim(0, np.max(ts_L2))

    if args.saveflag:
        savedir = args.savedir
        path = os.path.join(savedir, "lyapunov_orbit.png")
        plt.tight_layout()
        plt.savefig(path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
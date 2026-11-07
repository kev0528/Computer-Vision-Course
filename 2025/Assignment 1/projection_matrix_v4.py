import numpy as np
import cv2

# 建立 2N×12 矩陣 A 
def build_A(pts3d, pts2d):
    N = pts3d.shape[0]
    A = np.zeros((2 * N, 12))
    X, Y, Z = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
    u, v = pts2d[:, 0], pts2d[:, 1]
    for i in range(N):
        A[2*i]   = [X[i], Y[i], Z[i], 1, 0, 0, 0, 0,
                    -u[i]*X[i], -u[i]*Y[i], -u[i]*Z[i], -u[i]]
        A[2*i+1] = [0, 0, 0, 0, X[i], Y[i], Z[i], 1,
                    -v[i]*X[i], -v[i]*Y[i], -v[i]*Z[i], -v[i]]
    return A

# 2D 點正規化 
def normalize_points_2d(pts2d):
    mu = pts2d.mean(axis=0)
    d  = np.linalg.norm(pts2d - mu, axis=1).mean()
    s  = np.sqrt(2) / d
    T  = np.array([[s, 0, -s*mu[0]],
                   [0, s, -s*mu[1]],
                   [0, 0, 1]], dtype=float)
    xh = np.hstack([pts2d, np.ones((pts2d.shape[0], 1))])
    xn = (T @ xh.T).T
    return xn[:, :2] / xn[:, 2:3], T

#  3D 點正規化 
def normalize_points_3d(pts3d):
    mu = pts3d.mean(axis=0)
    d  = np.linalg.norm(pts3d - mu, axis=1).mean()
    s  = np.sqrt(3) / d
    U  = np.array([[s, 0, 0, -s*mu[0]],
                   [0, s, 0, -s*mu[1]],
                   [0, 0, s, -s*mu[2]],
                   [0, 0, 0, 1]], dtype=float)
    Xh = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])
    Xn = (U @ Xh.T).T
    return Xn[:, :3] / Xn[:, 3:4], U

# Normalized 投影矩陣
def projection_matrix_normalized(pts3d, pts2d):
    pts2d_n, T = normalize_points_2d(pts2d)
    pts3d_n, U = normalize_points_3d(pts3d)
    A = build_A(pts3d_n, pts2d_n)
    _, _, Vt = np.linalg.svd(A)
    Pn = Vt[-1, :].reshape(3, 4)
    P  = np.linalg.inv(T) @ Pn @ U
    if abs(P[-1, -1]) > 1e-12:
        P /= P[-1, -1]
    return P

# 從 P 取得 K 
def K_from_P(P: np.ndarray) -> np.ndarray:
    K, R, tvec, *_ = cv2.decomposeProjectionMatrix(P)
    K = (K / K[2, 2]).astype(float)
    s = np.sign(np.diag(K))
    s[s == 0] = 1.0
    D = np.diag(s)
    K = K @ D
    K[np.tril_indices(3, k=-1)] = 0.0
    K /= K[2, 2]
    K[0,1] = 0.0 # s indicates the skew ; typically s = 0
    return K

# 從 P 取得 R 
def R_from_P(P: np.ndarray) -> np.ndarray:
    K, R, tvec, *_ = cv2.decomposeProjectionMatrix(P)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R *= -1
    return R.astype(float)

# 從 P 取得 t 
def t_from_P(P: np.ndarray) -> np.ndarray:
    K, R, tvec, *_ = cv2.decomposeProjectionMatrix(P)

    # 相機中心 C 
    C = (tvec[:3] / tvec[3]).reshape(3)

    # t = -R * C
    t = -R @ C
    return t.astype(float)

#  用投影矩陣計算投影後的 2D 點 
def project_points(P, pts3d):

    # 齊次座標
    Xh = np.hstack([pts3d, np.ones((pts3d.shape[0], 1))])

    # 投影
    xh = (P @ Xh.T).T

    # 齊次除法
    uv = xh[:, :2] / xh[:, 2:3]
    return uv

# 比較投影結果與真實2D點，計算平均誤差 (pixel)
def average_projection_error(P, pts3d, pts2d_true):
    proj_2d = project_points(P, pts3d)
    errors = np.linalg.norm(proj_2d - pts2d_true, axis=1)
    avg_error = np.mean(errors)
    return avg_error

if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)

    # 讀取2D、3D點的資料
    pts2d = np.loadtxt("Pt2D.txt")
    pts3d = np.loadtxt("Pt3D.txt")

    P = projection_matrix_normalized(pts3d, pts2d)
    print("P (Normalized):\n", P)

    K = K_from_P(P)
    R = R_from_P(P)
    t = t_from_P(P)

    print("K:\n", K)
    print("R:\n", R)
    print("t:\n", t)

     # 計算平均誤差
    avg_err = average_projection_error(P, pts3d, pts2d)
    print("Average projection error: %.6f pixels" % avg_err)
    


import numpy as np
import math

class JacobianMetrics:
    def __init__(self, robot_config_file="ur3e.yml", tcp_link="tool0"):
        self.tcp_link = tcp_link

    def compute_dh_jacobian(self, q: np.ndarray):
        """
        Compute analytic Jacobian using standard UR3e DH parameters.
        q: [B, 6] numpy array (in radians)
        """
        # UR3e DH parameters
        d = [0.15185, 0.0, 0.0, 0.13105, 0.08535, 0.0921]
        a = [0.0, -0.24355, -0.2132, 0.0, 0.0, 0.0]
        alpha = [math.pi/2, 0.0, 0.0, math.pi/2, -math.pi/2, 0.0]
        
        B = q.shape[0]
        J_dh = np.zeros((B, 6, 6), dtype=np.float32)
        
        for b in range(B):
            qb = q[b]
            z = []
            p = []
            T = np.eye(4, dtype=np.float32)
            
            z.append(T[0:3, 2].copy())
            p.append(T[0:3, 3].copy())
            
            for i in range(6):
                ct = math.cos(qb[i])
                st = math.sin(qb[i])
                ca = math.cos(alpha[i])
                sa = math.sin(alpha[i])
                
                Ti = np.array([
                    [ct, -st*ca,  st*sa, a[i]*ct],
                    [st,  ct*ca, -ct*sa, a[i]*st],
                    [0,   sa,     ca,    d[i]],
                    [0,   0,      0,     1]
                ], dtype=np.float32)
                
                T = np.matmul(T, Ti)
                z.append(T[0:3, 2].copy())
                p.append(T[0:3, 3].copy())
                
            p_n = p[-1]
            for i in range(6):
                z_i = z[i]
                p_i = p[i]
                # J_v = z_i x (p_n - p_i)
                J_v = np.cross(z_i, p_n - p_i)
                # J_w = z_i
                J_w = z_i
                J_dh[b, 0:3, i] = J_v
                J_dh[b, 3:6, i] = J_w
                
        return J_dh

    def get_metrics(self, q: np.ndarray):
        """
        Compute metrics: sigma_min, manipulability (w), condition_number (cond)
        q: [B, 6] numpy array
        Returns dict with arrays of shape [B]
        """
        J = self.compute_dh_jacobian(q)
        
        # SVD: J = U * S * V^T
        U, S, Vh = np.linalg.svd(J)
        
        sigma_min = S[:, -1]
        sigma_max = S[:, 0]
        
        # Manipulability w = sqrt(det(J * J^T)) = product of singular values
        w = np.prod(S, axis=1)
        
        cond = sigma_max / (sigma_min + 1e-9)
        
        return {
            "sigma_min": sigma_min,
            "manipulability": w,
            "condition_number": cond,
            "singular_values": S
        }

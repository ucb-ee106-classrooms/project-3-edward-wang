import time 
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import approx_fprime
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 14


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][1] is the thrust of the quadrotor
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is translational position in x (m),
            x[i][1] is translational position in z (m),
            x[i][2] is the bearing (rad) of the quadrotor
            x[i][3] is translational velocity in x (m/s),
            x[i][4] is translational velocity in z (m/s),
            x[i][5] is angular velocity (rad/s),
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][1] is distance to the landmark (m)
            y[i][2] is relative bearing (rad) w.r.t. the landmark
        x_hat : list
            A list of estimated system states. It should follow the same format
            as x.
        dt : float
            Update frequency of the estimator.
        fig : Figure
            matplotlib Figure for real-time plotting.
        axd : dict
            A dictionary of matplotlib Axis for real-time plotting.
        ln* : Line
            matplotlib Line object for ground truth states.
        ln_*_hat : Line
            matplotlib Line object for estimated states.
        canvas_title : str
            Title of the real-time plot, which is chosen to be estimator type.

    Notes
    ----------
        The landmark is positioned at (0, 5, 5).
    """
    # noinspection PyTypeChecker
    def __init__(self, is_noisy=False):
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.t = []
        self.fig, self.axd = plt.subplot_mosaic(
            [['xz', 'phi'],
             ['xz', 'x'],
             ['xz', 'z']], figsize=(20.0, 10.0))
        self.ln_xz, = self.axd['xz'].plot([], linewidth = 2, label='True')
        self.ln_xz_hat, = self.axd['xz'].plot([], linewidth=2, label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_z, = self.axd['z'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_z_hat, = self.axd['z'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'

        # Defined in dynamics.py for the dynamics model
        # m is the mass and J is the moment of inertia of the quadrotor 
        self.gr = 9.81 
        self.m = 0.92
        self.J = 0.0023
        # These are the X, Y, Z coordinates of the landmark
        self.landmark = (0, 5, 5)

        # This is a (N,12) where it's time, x, u, then y_obs 
        if is_noisy:
            with open('noisy_data.npy', 'rb') as f:
                self.data = np.load(f)
        else:
            with open('data.npy', 'rb') as f:
                self.data = np.load(f)

        self.dt = self.data[-1][0]/self.data.shape[0]


    def run(self):
        step_times = []  # To store runtime for each step
        for i, data in enumerate(self.data):
            step_start = time.perf_counter()  # Start timer
            self.t.append(np.array(data[0]))
            self.x.append(np.array(data[1:7]))
            self.u.append(np.array(data[7:9]))
            self.y.append(np.array(data[9:12]))
            if i == 0:
                self.x_hat.append(self.x[-1])
            else:
                self.update(i)
            
            # compute run time
            step_end = time.perf_counter()  # End timer
            elapsed = step_end - step_start
            step_times.append(elapsed)

        # compute average run time
        avg_time = sum(step_times) / len(step_times)
        print(f"Average runtime per step: {avg_time:.10f} seconds")
        
        # plot compute time
        # plt.figure(figsize=(10, 5))
        # plt.plot(range(len(step_times)), step_times, marker='o', linestyle='-')
        # plt.xlabel("Timestep")
        # plt.ylabel("Runtime (s)")
        # plt.title("Runtime per Timestep for Extended Kalman Filter on Planar Quadrotor")
        # plt.grid(True)
        # plt.show()

        # compute tracking error
        errors = []
        for true_state, estimated_state in zip(self.x, self.x_hat):
            # Extracting only the x and z positions (indices 0 and 1)
            pos_true = np.array(true_state[:2])
            pos_est = np.array(estimated_state[:2])
            error = np.linalg.norm(pos_true - pos_est)
            errors.append(error)
        avg_error = sum(errors) / len(errors)
        max_error = max(errors)
        print(f"Average state error per step: {avg_error:.5f} meters")
        print(f"Max state error: {max_error:.5f} meters")

        # plot track error
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(errors)), errors, marker='o', linestyle='-')
        plt.xlabel("Timestep")
        plt.ylabel("coordinate tracking error (m)")
        plt.title("tracking error for Extended Kalman Filter on Planar Quadrotor")
        plt.grid(True)
        plt.show()


        return self.x_hat

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xz'].set_title(self.canvas_title)
        self.axd['xz'].set_xlabel('x (m)')
        self.axd['xz'].set_ylabel('z (m)')
        self.axd['xz'].set_aspect('equal', adjustable='box')
        self.axd['xz'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].set_xlabel('t (s)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].set_xlabel('t (s)')
        self.axd['x'].legend()
        self.axd['z'].set_ylabel('z (m)')
        self.axd['z'].set_xlabel('t (s)')
        self.axd['z'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xzline(self.ln_xz, self.x)
        self.plot_xzline(self.ln_xz_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_zline(self.ln_z, self.x)
        self.plot_zline(self.ln_z_hat, self.x_hat)

    def plot_xzline(self, ln, data):
        if len(data):
            x = [d[0] for d in data]
            z = [d[1] for d in data]
            ln.set_data(x, z)
            self.resize_lim(self.axd['xz'], x, z)

    def plot_philine(self, ln, data):
        if len(data):
            t = self.t
            phi = [d[2] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = self.t
            x = [d[0] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_zline(self, ln, data):
        if len(data):
            t = self.t
            z = [d[1] for d in data]
            ln.set_data(t, z)
            self.resize_lim(self.axd['z'], t, z)

    # noinspection PyMethodMayBeStatic
    def resize_lim(self, ax, x, y):
        xlim = ax.get_xlim()
        ax.set_xlim([min(min(x) * 1.05, xlim[0]), max(max(x) * 1.05, xlim[1])])
        ylim = ax.get_ylim()
        ax.set_ylim([min(min(y) * 1.05, ylim[0]), max(max(y) * 1.05, ylim[1])])

class OracleObserver(Estimator):
    """Oracle observer which has access to the true state.

    This class is intended as a bare minimum example for you to understand how
    to work with the code.

    Example
    ----------
    To run the oracle observer:
        $ python drone_estimator_node.py --estimator oracle_observer
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Oracle Observer'

    def update(self, _):
        self.x_hat.append(self.x[-1])


class DeadReckoning(Estimator):
    """Dead reckoning estimator.

    Your task is to implement the update method of this class using only the
    u attribute and x0. You will need to build a model of the unicycle model
    with the parameters provided to you in the lab doc. After building the
    model, use the provided inputs to estimate system state over time.

    The method should closely predict the state evolution if the system is
    free of noise. You may use this knowledge to verify your implementation.

    Example
    ----------
    To run dead reckoning:
        $ python drone_estimator_node.py --estimator dead_reckoning
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Dead Reckoning'

    def update(self, _):
        if len(self.x_hat) > 0:
            # TODO: Your implementation goes here!
            # You may ONLY use self.u and self.x[0] for estimation
            x_prev = self.x_hat[-1]
            u_prev = self.u[-1]

            self.Bd = np.array([x_prev[3],x_prev[4],x_prev[5],0,-self.gr,0])
            self.Cd = np.array([[0,0],
                               [0,0],
                               [0,0],
                               [-np.sin(x_prev[2])/self.m,0],
                               [np.cos(x_prev[2])/self.m,0],
                               [0,1/self.J]])
              
            x_pred = x_prev + self.dt * (self.Bd + self.Cd @ u_prev)

            self.x_hat.append(x_pred)

            # raise NotImplementedError

# noinspection PyPep8Naming
class ExtendedKalmanFilter(Estimator):
    """Extended Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    unicycle model and linearize it at every operating point. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive extended Kalman filter update rule.

    Hint: You may want to reuse your code from DeadReckoning class and
    KalmanFilter class.

    Attributes:
    ----------
        landmark : tuple
            A tuple of the coordinates of the landmark.
            landmark[0] is the x coordinate.
            landmark[1] is the y coordinate.
            landmark[2] is the z coordinate.

    Example
    ----------
    To run the extended Kalman filter:
        $ python drone_estimator_node.py --estimator extended_kalman_filter
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Extended Kalman Filter'
        # TODO: Your implementation goes here!
        # You may define the Q, R, and P matrices below.
        
        self.Q = np.eye(6)
        self.R = np.eye(2) * 100
        self.P = np.eye(6)
        self.xL = 0
        self.yL = 5
        self.zL = 5

    # noinspection DuplicatedCode
    def update(self, i):
        if len(self.x_hat) > 0: #and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation

            #process data
            u_prev = np.array(self.u[-1])
            x_prev = np.array(self.x_hat[-1])
            y_mea = np.array(self.y[-1])

            # prediction -- state
            x_pred = self.g(self.x_hat[-1],self.u[-1])

            # prediction -- variance
            self.A = self.approx_A(x_prev, u_prev)
            P_pred = self.A @ self.P @ self.A.T + self.Q

            # Kalman gain calculation
            self.C = self.approx_C(x_pred)
            K = P_pred @ self.C.T @ np.linalg.inv(self.C @ P_pred @ self.C.T + self.R)

            # update -- state
            X_mea = x_pred + K @ (y_mea - self.h(x_pred, y_mea))

            # update -- variance
            P_mea = P_pred - K @ self.C @ P_pred

            self.x_hat.append(X_mea)
            self.P = P_mea
            # raise NotImplementedError

    def g(self, x, u):
            self.Bd = np.array([x[3],x[4],x[5],0,-self.gr,0])
            self.Cd = np.array([[0,0],
                                [0,0],
                                [0,0],
                                [-np.sin(x[2])/self.m,0],
                                [np.cos(x[2])/self.m,0],
                                [0,1/self.J]])
              
            return x + self.dt * (self.Bd + self.Cd @ u)
        # raise NotImplementedError

    def h(self, x, y_obs):
        dx = self.xL - x[0]
        dz = self.zL - x[1]
        r = np.sqrt(dx**2 + self.yL**2 + dz**2)
        bearing = x[2]
        return np.array([r, bearing])
        # raise NotImplementedError

    def approx_A(self, x, u):
        return np.array([[1,0,0,self.dt,0,0],
                          [0,1,0,0,self.dt,0],
                          [0,0,1,0,0,self.dt],
                          [0,0,-self.dt*u[0]*np.cos(x[2])/self.m,1,0,0],
                          [0,0,-self.dt*u[0]*np.sin(x[2])/self.m,0,1,0],
                          [0,0,0,0,0,1]])
    
    def approx_C(self, x):
        # raise NotImplementedError
        # Use an epsilon based on machine precision for finite differences
        # epsilon = np.sqrt(np.finfo(float).eps)
        # jac = np.zeros((2,6))
        # for i in range(2):
        #     def f_i(x,i=i):
        #         return self.h(x, y_mea)[i]
        #     jac[i, :] = approx_fprime(x, f_i, epsilon)
        # return jac
        
        dx = self.xL - x[0]
        dz = self.zL - x[1]
        r = np.sqrt(dx**2 + self.yL**2 + dz**2)

        return np.array([[-dx/r,-dz/r,0,0,0,0],
                         [0,0,1,0,0,0]])

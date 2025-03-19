import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['FreeSans', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 14


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        d : float
            Half of the track width (m) of TurtleBot3 Burger.
        r : float
            Wheel radius (m) of the TurtleBot3 Burger.
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][0] is timestamp (s),
            u[i][1] is left wheel rotational speed (rad/s), and
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is timestamp (s),
            x[i][1] is bearing (rad),
            x[i][2] is translational position in x (m),
            x[i][3] is translational position in y (m),
            x[i][4] is left wheel rotational position (rad), and
            x[i][5] is right wheel rotational position (rad).
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][0] is timestamp (s),
            y[i][1] is translational position in x (m) when freeze_bearing:=true,
            y[i][1] is distance to the landmark (m) when freeze_bearing:=false,
            y[i][2] is translational position in y (m) when freeze_bearing:=true, and
            y[i][2] is relative bearing (rad) w.r.t. the landmark when
            freeze_bearing:=false.
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
        sub_u : rospy.Subscriber
            ROS subscriber for system inputs.
        sub_x : rospy.Subscriber
            ROS subscriber for system states.
        sub_y : rospy.Subscriber
            ROS subscriber for system outputs.
        tmr_update : rospy.Timer
            ROS Timer for periodically invoking the estimator's update method.

    Notes
    ----------
        The frozen bearing is pi/4 and the landmark is positioned at (0.5, 0.5).
    """
    # noinspection PyTypeChecker
    def __init__(self):
        self.d = 0.08
        self.r = 0.033
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        # self.x_hat_kf1 = [] 
        # self.x_hat_kf2 = [] 
        # self.p_hat = [] # store estimation
        self.dt = 0.1
        self.fig, self.axd = plt.subplot_mosaic(
            [['xy', 'phi'],
             ['xy', 'x'],
             ['xy', 'y'],
             ['xy', 'thl'],
             ['xy', 'thr']], figsize=(20.0, 10.0))
        self.ln_xy, = self.axd['xy'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xy_hat, = self.axd['xy'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_y, = self.axd['y'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_y_hat, = self.axd['y'].plot([], 'o-c', label='Estimated')
        self.ln_thl, = self.axd['thl'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thl_hat, = self.axd['thl'].plot([], 'o-c', label='Estimated')
        self.ln_thr, = self.axd['thr'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thr_hat, = self.axd['thr'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'
        self.sub_u = rospy.Subscriber('u', Float32MultiArray, self.callback_u)
        self.sub_x = rospy.Subscriber('x', Float32MultiArray, self.callback_x)
        self.sub_y = rospy.Subscriber('y', Float32MultiArray, self.callback_y)
        self.tmr_update = rospy.Timer(rospy.Duration(self.dt), self.update)

        # # NEW: Create a second figure for distance error
        # self.fig_dist, self.ax_dist = plt.subplots()
        # # Create multiple line objects, one per filter
        # self.ln_dist_kf1, = self.ax_dist.plot([], [], 'o-b', label='KF1 Error')
        # self.ln_dist_kf2, = self.ax_dist.plot([], [], 'o-r', label='KF2 Error')
        # self.ax_dist.set_title('Distance Error Comparison')
        # self.ax_dist.set_xlabel('Time (s)')
        # self.ax_dist.set_ylabel('Distance Error (m)')
        # self.ax_dist.legend()


    def callback_u(self, msg):
        self.u.append(msg.data)

    def callback_x(self, msg):
        self.x.append(msg.data)
        if len(self.x_hat) == 0:
            self.x_hat.append(msg.data)

    def callback_y(self, msg):
        self.y.append(msg.data)

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xy'].set_title(self.canvas_title)
        self.axd['xy'].set_xlabel('x (m)')
        self.axd['xy'].set_ylabel('y (m)')
        self.axd['xy'].set_aspect('equal', adjustable='box')
        self.axd['xy'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].legend()
        self.axd['y'].set_ylabel('y (m)')
        self.axd['y'].legend()
        self.axd['thl'].set_ylabel('theta L (rad)')
        self.axd['thl'].legend()
        self.axd['thr'].set_ylabel('theta R (rad)')
        self.axd['thr'].set_xlabel('Time (s)')
        self.axd['thr'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xyline(self.ln_xy, self.x)
        self.plot_xyline(self.ln_xy_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_yline(self.ln_y, self.x)
        self.plot_yline(self.ln_y_hat, self.x_hat)
        self.plot_thlline(self.ln_thl, self.x)
        self.plot_thlline(self.ln_thl_hat, self.x_hat)
        self.plot_thrline(self.ln_thr, self.x)
        self.plot_thrline(self.ln_thr_hat, self.x_hat)

        # self.plot_dist_error(self.ln_dist_kf1, self.x_hat_kf1)
        # self.plot_dist_error(self.ln_dist_kf2, self.x_hat_kf2)

        # # Redraw
        # plt.draw()
        # plt.pause(0.001)

    def plot_dist_error(self, ln, x_hat_list):
        """
        Generic helper that computes distance error vs time
        for a given set of estimates x_hat_list.
        ln is the line handle to update.
        """
        # We need both ground truth (self.x) and the estimates
        N = min(len(self.x), len(x_hat_list))
        if N == 0:
            return

        t_vals = []
        dist_vals = []
        for i in range(N):
            t_vals.append(self.x[i][0])       # time from ground truth
            x_true = self.x[i][2]
            y_true = self.x[i][3]
            x_est = x_hat_list[i][2]
            y_est = x_hat_list[i][3]
            dist = np.sqrt((x_true - x_est)**2 + (y_true - y_est)**2)
            dist_vals.append(dist)

        ln.set_data(t_vals, dist_vals)
        self.ax_dist.relim()
        self.ax_dist.autoscale_view()

    def plot_xyline(self, ln, data):
        if len(data):
            x = [d[2] for d in data]
            y = [d[3] for d in data]
            ln.set_data(x, y)
            self.resize_lim(self.axd['xy'], x, y)

    def plot_philine(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            phi = [d[1] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            x = [d[2] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_yline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            y = [d[3] for d in data]
            ln.set_data(t, y)
            self.resize_lim(self.axd['y'], t, y)

    def plot_thlline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thl = [d[4] for d in data]
            ln.set_data(t, thl)
            self.resize_lim(self.axd['thl'], t, thl)

    def plot_thrline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thr = [d[5] for d in data]
            ln.set_data(t, thr)
            self.resize_lim(self.axd['thr'], t, thr)

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
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=oracle_observer \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
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
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=dead_reckoning \
            noise_injection:=true \
            freeze_bearing:=false
    For debugging, you can simulate a noise-free unicycle model by setting
    noise_injection:=false.
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Dead Reckoning'

    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may ONLY use self.u and self.x[0] for estimation

            t_prev, phi_prev, x_prev, y_prev, thetaL_prev, thetaR_prev = self.x_hat[-1]

            # Current control input: [time, omega_L, omega_R]
            _, omega_L, omega_R = self.u[-1]

            # Compute time step
            dt = self.x[-1][0] - t_prev

            # Unicycle model parameters
            r = self.r
            d = self.d

            # ----- State Propagation -----
            # 1) Heading
            phi_new = phi_prev + (r / (2.0 * d)) * (omega_R - omega_L) * dt
            x_new = x_prev + (r / 2.0) * (omega_L + omega_R) * np.cos(phi_prev) * dt
            y_new = y_prev + (r / 2.0) * (omega_L + omega_R) * np.sin(phi_prev) * dt
            thetaL_new = thetaL_prev + omega_L * dt
            thetaR_new = thetaR_prev + omega_R * dt

            # Optionally wrap heading to [-pi, pi] if desired
            # phi_new = (phi_new + np.pi) % (2.0 * np.pi) - np.pi

            # Append new estimate: [time, phi, x, y, theta_L, theta_R]
            self.x_hat.append([
                self.x[-1][0],
                phi_new,
                x_new,
                y_new,
                thetaL_new,
                thetaR_new
            ])

            # raise NotImplementedError


class KalmanFilter(Estimator):
    """Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    linear unicycle model at the default bearing of pi/4. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive Kalman filter update rule.

    Attributes:
    ----------
        phid : float
            Default bearing of the turtlebot fixed at pi / 4.

    Example
    ----------
    To run the Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=kalman_filter \
            noise_injection:=true \
            freeze_bearing:=true
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Kalman Filter'
        self.phid = np.pi / 4
        # TODO: Your implementation goes here!
        # You may define the A, C, Q, R, and P matrices below.
        
        phid = self.phid
        r = self.r
        d = self.d
        
        self.A = np.eye(5)
        self.C = np.array([[0,1,0,0,0],
                           [0,0,1,0,0]])
        self.Q = np.eye(5) # process noise
        self.R = np.eye(2)*10 # measurement noise
        self.P = np.eye(5)

    # noinspection DuplicatedCode
    # noinspection PyPep8Naming
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation

            # define parameters 
            phid = self.phid
            r = self.r
            d = self.d

            # process data
            t_prev, phi_prev, x_prev, y_prev, thetaL_prev, thetaR_prev = self.x_hat[-1]
            _, omega_L, omega_R = self.u[-1] # Current control input: [time, omega_L, omega_R]
            dt = self.x[-1][0] - t_prev # Compute time step

            self.B = dt * np.array([[-r/(2*d),r/(2*d)],
                                [r*np.cos(phid)/2,r*np.cos(phid)/2],
                                [r*np.sin(phid)/2,r*np.sin(phid)/2],
                                [1,0],
                                [0,1]])

            x_prev = np.array([phi_prev, x_prev, y_prev, thetaL_prev, thetaR_prev])
            u_prev = np.array([omega_L, omega_R])
            y_cur = np.array([self.y[-1][1],self.y[-1][2]])

            # print(x_prev,u_prev)

            # prediction -- state
            x_pred = self.A @ x_prev + self.B @ u_prev

            # prediction -- variance
            P_pred = self.A @ self.P @ self.A.T + self.Q

            # Compute Kalman gain
            K = P_pred @ self.C.T @ np.linalg.inv(self.C @ P_pred @ self.C.T + self.R)

            # update -- state
            X_mea = x_pred + K @ (y_cur-self.C @ x_pred)

            # update -- variance 
            P_mea = P_pred - K @ self.C @ P_pred

            # store step update
            self.x_hat.append([self.x[-1][0],*X_mea])
            self.P = P_mea

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

    Example
    ----------
    To run the extended Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=extended_kalman_filter \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Extended Kalman Filter'
        self.landmark = (0.5, 0.5)
        # TODO: Your implementation goes here!
        # You may define the Q, R, and P matrices below.

    # noinspection DuplicatedCode
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation
            raise NotImplementedError


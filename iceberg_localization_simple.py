######################################
# Taylor Howell
# State Estimation (AA273 Spring 2018)
# June 12, 2018
######################################

import numpy as np
import matplotlib.pyplot as plt
import imageio
import time
from scipy.stats import multivariate_normal


class model():
    def __init__(self, Map):
        self.dt = 0.5  # discretized time step
        self.n = 3  # number of states: [x_position, y_position, heading]'
        self.m = 2  # number of measurements: [range, bearing]'
        self.v = 10  # forward velocity of robot (m/s)
        self.min_standoff = 50  # minimum distance between robot and body
        self.max_standoff = 100  # maximum distance between robot and body
        self.measurement_limit = 3  # maximum number of measurements than can be processed at each time step
        self.Q = 0.001 * np.eye(self.n) * self.dt  # process covariance
        self.R = 50 * np.eye(self.m)  # measurement covariance

        self.Map = Map  # map matrix: 2 x (number of map features)

    def dynamics(self, x, u, noise=True):
        # Dubins path dynamics
        x[0, 0] += self.dt * u[0, 0] * np.cos(x[2, 0])
        x[1, 0] += self.dt * u[0, 0] * np.sin(x[2, 0])
        x[2, 0] += self.dt * u[1, 0]

        # additive Gaussian white process noise (only on robot dynamics, not body states)
        if noise:
            noise_ = self.gw_noise(self.Q)
            x[0, 0] += noise_[0, 0]
            x[1, 0] += noise_[1, 0]
            x[2, 0] += noise_[2, 0]

        return x.reshape((self.n, 1))

    def measurement(self, x, feature, noise=True):
        y = np.zeros((self.m, 1))
        # range measurement
        y[0, 0] = np.linalg.norm(feature - x[0:2, :])
        # bearing measurements
        y[1, 0] = np.arctan2((feature[1, 0] - x[1, 0]), (feature[0, 0] - x[0, 0])) - x[2, 0]

        # additive Gaussian white measurement noise
        if noise:
            y += self.gw_noise(self.R)

        return y

    def control(self, V, mode=0, error=0, error_prev=0, error_array=np.array([0])):
        Kp = 0.01
        Kd = 0.1
        Ki = 0.0001
        # straight mode
        if mode == 0:
            # drive straight
            return np.array([[5 * V], [0]])

        # circling mode (PID)
        if mode == 1:
            PID = Kp * error + Kd * (error - error_prev) / self.dt + Ki * np.sum(error)
            return np.array([[V], [PID]])

    def gw_noise(self, cov):
        # Gaussian white noise
        if isinstance(cov, np.ndarray):
            return np.random.multivariate_normal(np.zeros(cov.shape[0], ), cov).reshape((cov.shape[0], 1))
        else:
            return np.random.normal(0, np.sqrt(cov))


class body():
    def __init__(self, n=20, r=60, var=3):
        self.n = n  # number of sides on the body
        self.r = r  # (m) 'average' radius of the body
        self.var = var  # measure or 'jaggedness' in the body
        self.feature_dim = 2
        self.polygon = self.gen_polygon(self.n, self.r, self.var)  # generate 2D polygon representation of body in polar coordinates [r,theta]

        # transform map into cartesian
        self.Map = np.zeros((self.feature_dim, self.n))
        x, y = self.polar2cartesian(self.polygon[0, :], self.polygon[1, :])
        self.Map[0, :] = x.ravel()
        self.Map[1, :] = y.ravel()

    def gen_polygon(self, n, r, var=1):
        # generates a random n-sided polygon with an average radius r
        p = np.zeros((self.feature_dim, self.n))
        radius = np.random.normal(r, var, n).ravel()
        angles = [np.random.uniform(0, 2 * np.pi) for q in range(self.n)]
        angles.sort()
        p[0, :] = radius
        p[1, :] = angles
        return p

    def polar2cartesian(self, r, theta):
        x = np.multiply(r, np.cos(theta))
        y = np.multiply(r, np.sin(theta))
        return x, y

    def cartesian2polar(self, x, y):
        r = np.sqrt(np.multiply(x, x) + np.multiply(y, y))
        theta = np.arctan2(y, x)
        return r, theta

    def plot_polygon(self):
        x, y = self.polar2cartesian(self.polygon[0, :], self.polygon[1, :])
        plt.plot(x, y, 'k.-')
        plt.plot((x[-1], x[0]), (y[-1], y[0]), 'ko.-')


class PF():
    def __init__(self, model, body, N, mu0, sigma0, random_percent=0.1):
        self.model = model
        self.body = body
        self.N = N  # number of particles
        self.random_percent = random_percent
        self.N_random = int(self.N * random_percent)  # number of particles to randomly sample during resampling
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.particles = self.particle_initialization(mu0, sigma0)

    def particle_initialization(self, mu, sigma, mode=0):
        # initialize particles in circle around body
        if mode == 1:
            print('PF initialized in circle around body')
            radius = (self.body.r + np.mean([self.model.min_standoff, self.model.max_standoff])) * np.ones(self.N, )
            angle = np.linspace(0, 2 * np.pi, self.N)
            x, y = self.body.polar2cartesian(radius, angle)
            P = np.random.multivariate_normal(mu.ravel(), sigma, self.N).T
            P[0, :] = x
            P[1, :] = y
            return P

        # initialize particles randomly around body
        if mode == 2:
            print('PF initialized in randomized circle around body')
            radius = np.random.normal(self.body.r + np.mean([self.model.min_standoff, self.model.max_standoff]), 25,
                                      self.N).ravel()
            angles = [np.random.uniform(0, 2 * np.pi) for q in range(self.N)]
            # print('radius: {}'.format(radius))
            # print('angles: {}'.format(angles))
            x, y = self.body.polar2cartesian(radius, angles)
            P = np.random.multivariate_normal(mu.ravel(), sigma, self.N).T
            P[0, :] = x
            P[1, :] = y
            return P

        # initialize particles (randomized) uniformly around body
        if mode == 3:
            print('PF initialized from uniform random distribution around body')
            P = np.random.multivariate_normal(mu.ravel(), sigma, self.N).T
            r = self.body.r + self.model.max_standoff
            P[0, :] = np.random.uniform(-r, r, self.N)
            P[1, :] = np.random.uniform(-r, r, self.N)
            return P

    def prediction_update(self, mu, u):
        mu_ = np.zeros(mu.shape)
        for i in range(self.N):
            mu_[:, i] = self.model.dynamics(mu[:, i:i + 1], u, noise=True).ravel()
        return mu_

    def measurement_update(self, mu_, Y, idx, t, tf):
        w_ = np.ones(self.N, )

        for i in range(self.N):
            for j in idx:
                y = Y[:, j:j + 1]
                y_hat = self.model.measurement(mu_[:, i:i + 1], self.body.Map[:, j:j + 1], noise=False)
                p = multivariate_normal.pdf(y.ravel(), mean=y_hat.ravel(), cov=self.model.R)
                w_[i] *= p
                if p == 0:
                    print('p = 0: Fix measurement covariance R')  # check for bad weights, readjust R

        w = w_ / np.sum(w_)  # normalize weights

        # resample
        nr = int(self.N_random * (tf - t) / tf)
        # print('nr: {}'.format(nr))
        mu = np.zeros(mu_.shape)
        for i in range(self.N - nr):
            # print('resample: {}'.format(i))
            idx_random = np.random.choice(range(self.N), 1, p=w)
            mu[:, i] = mu_[:, int(idx_random)].ravel()

        # resample randoms
        maxIdx = w.argsort()[-1:][::-1]
        # print('maxIdx: {}'.format(maxIdx))
        # print(mu_[:,int(maxIdx):int(maxIdx)+1])
        mu[:, self.N - nr:self.N] = np.random.multivariate_normal(mu_[:, int(maxIdx):int(maxIdx) + 1].ravel(),
                                                                  self.sigma0, nr).T
        # print('mu: \n {}'.format(mu))
        return mu


class simulator():
    def __init__(self, model, body, x0, tf):
        self.model = model
        self.body = body
        self.x0 = x0  # initial state
        self.t = 0
        self.tf = tf  # total simulation time
        self.mode = 0  # modes: 0 = travel straight to body; 1 = circle the body
        self.K = 2  # mode transition index

    def run_dynamics(self):
        print('*Dynamics Simulation*: STARTED')
        n_steps = int(np.floor(self.tf / self.model.dt))  # number of simulation time steps
        self.r_desired = self.body.r + np.mean([self.model.min_standoff, self.model.max_standoff])

        # initialize memory for results
        T = np.zeros(n_steps, )
        T[0] = self.t
        X = np.zeros((self.model.n, n_steps))
        X[:, 0] = self.x0.ravel()
        X_no_measurement = np.zeros((self.model.n, n_steps))
        X_no_measurement[:, 0] = self.x0.ravel()
        control = np.zeros((2, n_steps))
        results = dict()
        error = np.zeros(n_steps, )
        self.offset_desired = np.mean([self.model.min_standoff, self.model.max_standoff])  # desired radius to circle body

        # run dynamics
        for k in range(1, n_steps):
            # increment time
            self.t += self.model.dt

            # initial measurement to determine control
            Y = np.zeros((self.body.feature_dim, self.body.n))
            for j in range(self.body.n):
                Y[:, j] = self.model.measurement(X[:, k - 1:k], self.body.Map[:, j:j + 1]).ravel()
            y_min_range = min(Y[0, :])

            # control
            if self.mode == 0:
                # if robot is near body, change modes and start PID controller
                if y_min_range <= np.mean([self.model.min_standoff, self.model.max_standoff]):
                    print('---Arrived at body---')
                    self.mode = 1  # transition to 'wall-following'/circling mode
                    self.K = k
                # otherwise move straight
                else:
                    u = self.model.control(self.model.v, mode=self.mode)

            # circle the body mode:
            if self.mode == 1:
                u = self.model.control(self.model.v, mode=self.mode, error=error[k - 1], error_prev=error[k - 2], error_array=error[self.K:])

            control[:, k] = u.ravel()

            # increment robot
            X[:, k] = self.model.dynamics(X[:, k - 1:k], u).ravel()
            X_no_measurement[:, k] = self.model.dynamics(X_no_measurement[:, k - 1:k], u, noise=False).ravel()

            # compute trajectory error
            if self.mode == 0:
                error[k] = self.r_desired - np.sqrt(X_no_measurement[0, k] ** 2 + X_no_measurement[1, k] ** 2)  # idealized
            elif self.mode == 1:
                Y = np.zeros((self.body.feature_dim, self.body.n))
                for j in range(self.body.n):
                    Y[:, j] = self.model.measurement(X[:, k - 1:k], self.body.Map[:, j:j + 1]).ravel()
                y_min_range = min(Y[0, :])
                error[k] = self.offset_desired - y_min_range

            # cache values
            T[k] = self.t

            if k % 10 == 0 or k == n_steps:
                print('Time: {}/{}'.format(self.t, self.tf))

        results['T'] = T
        results['X'] = X
        results['X_no_measurement'] = X_no_measurement
        results['K'] = self.K  # transition index
        results['error'] = error
        results['n_steps'] = n_steps
        results['control'] = control
        results['map'] = self.body.Map
        print('*Dynamics Simulation*: COMPLETE')

        return results

    def run_filter(self, filter, results, initialization_mode=0):
        print('*Filter Simulation*: STARTED')
        MU = np.zeros((self.model.n, filter.N, results['n_steps']))
        MU[:, :, 0] = results['X_no_measurement'][:, 0:1].dot(np.ones((1, filter.N)))
        self.mode = 0
        for k in range(1, results['n_steps']):

            u = results['control'][:, k:k + 1]

            if k < results['K']:
                # haven't arrived at body, no measurements
                MU[:, :, k] = results['X_no_measurement'][:, k:k + 1].dot(np.ones((1, filter.N)))

            else:
                if k == results['K']:
                    print('---Arrived at body---')
                    MU[:, :, k - 1] = filter.particle_initialization(results['X_no_measurement'][:, k:k + 1], filter.sigma0, mode=initialization_mode)  # initialize particle filter using current odometry

                # prediction update
                mu_ = filter.prediction_update(MU[:, :, k - 1], u)

                # measurement update
                Y = np.zeros((self.body.feature_dim, self.body.n))
                for j in range(self.body.n):
                    Y[:, j] = self.model.measurement(results['X'][:, k:k + 1], self.body.Map[:, j:j + 1]).ravel()
                idx = Y[0, :].ravel().argsort()[:self.model.measurement_limit]  # only update using the closest measurements
                MU[:, :, k] = filter.measurement_update(mu_, Y, idx, results['T'][k],self.tf)  # measurement update for particles

            if k % 10 == 0 or k == results['n_steps']:
                print('Time: {}/{}'.format(results['T'][k], self.tf))

        results['MU'] = MU
        print('*Filtering Simulation*: COMPLETE')
        return results


def create_gif(results, name):
    # generates iceberg animation (.gif)
    print('.gif: GENERATING')
    plt.figure()
    labels = []
    for k in range(len(results['T'])):
        plt.clf()
        plot_trajectory(results, traj='X_no_measurement', k=k)  # plot odometry trajectory
        plot_map(results['map'])  # plot the body
        plot_PF(results, all=False, k=k)
        plot_trajectory(results, k=k)  # plot actual trajectory

        # plotting settings
        plt.gca().set_facecolor((96 / 255, 226 / 255, 230 / 255))  # set background to ocean blue
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.axis('equal')
        plt.axes().set_aspect('equal')

        plt.savefig('iceberg%i.png' % k)
        labels.append('iceberg%i.png' % k)

    with imageio.get_writer(name + '.gif', mode='I') as writer:
        for filename in labels:
            image = imageio.imread(filename)
            writer.append_data(image)
    writer.close()
    print('.gif: COMPLETE')


def plot_map(m, outline=False):
    if outline:
        plt.gca().add_patch(plt.Polygon(m.T, color=(0, 0, 0), fill=False))
    else:
        plt.gca().add_patch(plt.Polygon(m.T, color=(1, 1, 1)))


def plot_trajectory(results, traj='X', k=-1):
    plt.plot(results[traj][0, 0], results[traj][1, 0], 'r.')
    if traj == 'X':
        if k != -1:
            plt.plot(results[traj][0, :k + 1], results[traj][1, :k + 1], 'b', label='State')
            plt.plot(results[traj][0, k], results[traj][1, k], 'yo', label='Robot')
        else:
            plt.plot(results[traj][0, :], results[traj][1, :], 'b', label='State')
    else:
        if k != -1:
            plt.plot(results[traj][0, :k + 1], results[traj][1, :k + 1], 'c--', label='State')
        else:
            plt.plot(results[traj][0, :], results[traj][1, :], 'c--', label='State')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

def plot_PF(results, all=True, k=0):
    if all:
        f = lambda m, n: [i * n // m + n // (2 * m) for i in range(m)]
        B = f(15, len(results['T']))
        plt.plot(results['MU'][0, :, results['K'] - 1], results['MU'][1, :, results['K'] - 1], 'r.')
        for b in B:
            if b > results['K']:
                plt.plot(results['MU'][0, :, b], results['MU'][1, :, b], 'y.')
        plt.plot(results['MU'][0, :, -1], results['MU'][1, :, -1], 'g.')

    elif k >= results['K'] - 1:
        plt.plot(results['MU'][0, :, k], results['MU'][1, :, k], 'g.')


def plot_time_responses(results, i, j):
    # plot static results
    plt.figure()
    plot_trajectory(results, traj='X_no_measurement')
    plot_map(iceberg.Map, outline=True)
    plot_PF(results, all=True)
    plot_trajectory(results)
    plt.axis('equal')
    plt.axes().set_aspect('equal')
    plt.savefig('Iceberg_Trajectory_Estimation_Initialization_{}_N_{}.png'.format(i, j))
    # plt.show()

    # plot state estimates over time
    plt.figure()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.set_title('Robot State Estimates (Initialization: {}, N={})'.format(i, j))

    ax1.plot(results['T'], results['X'][0, :], 'b', label='State')
    ax1.plot(results['T'], [np.mean(results['MU'][0, :, q]) for q in range(len(results['T']))], 'r--', label='Estimate')
    ax1.axvline(x=results['T'][results['K'] - 1], color='g')
    ax1.set_ylabel('x')
    ax1.set_yticklabels([])
    ax1.legend()

    ax2.plot(results['T'], results['X'][1, :], 'b')
    ax2.plot(results['T'], [np.mean(results['MU'][1, :, q]) for q in range(len(results['T']))], 'r--')
    ax2.axvline(x=results['T'][results['K'] - 1], color='g')
    ax2.set_ylabel('y')

    ax3.plot(results['T'], results['X'][2, :], 'b')
    ax3.plot(results['T'], [np.mean(results['MU'][2, :, q]) for q in range(len(results['T']))], 'r--')
    ax3.axvline(x=results['T'][results['K'] - 1], color='g')
    ax3.set_ylabel('$\Theta$')
    ax3.set_xlabel('Time')
    plt.savefig('Robot_Time_Estimation_Initialization_{}_N_{}.png'.format(i, j))

if __name__ == "__main__":
    np.random.seed(0)
    tf = 120.0  # simulation time

    # body setup
    iceberg = body()

    # robot setup
    robot = model(iceberg.Map)
    x0 = np.array([[iceberg.r + np.mean([robot.min_standoff, robot.max_standoff])],
                   [0.],
                   [np.pi / 2.]])

    pos_start = np.array([[200], [-200]])  # initialize robot xy location
    x0[0:2] = pos_start
    x0[2] = np.arctan2(pos_start[1, 0], pos_start[0, 0]) + np.pi  # initialize robot heading toward body

    # run system dynamics with wall following
    sim = simulator(robot, iceberg, x0, tf)
    results = sim.run_dynamics()

    ## Filtering
    # initialization modes to test:
    mode = [1, 2, 3]  # odometry, circle, randomized circle, randomized uniform
    N_particles = [10, 100, 1000]  # number of particles

    # uncertainty in robot pose
    mu0 = x0.copy()
    sigma0 = np.eye(robot.n)

    for j in N_particles:
        for i in mode:
            # filter setup
            pf = PF(robot, iceberg, j, mu0, sigma0)

            # run filter
            start = time.time()
            results_filtered = sim.run_filter(pf, results, i)
            t = time.time() - start
            print('filter compute time: {}'.format(t))

            create_gif(results_filtered, 'Iceberg_Animation_Initialization_{}_N_{}'.format(i, j))
            plot_time_responses(results_filtered, i, j)

            # plt.figure()
            # plt.plot(results['T'],results['error'],) # plot error (off_setdesired - y_min) # check error for PID calibration

            plt.close('all')
    #plt.show()
    print('***All Simulations COMPLETE***')
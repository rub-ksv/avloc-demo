import numpy as np
import csv


class EKF(object):
    def __init__(self, process_noise_variance=1000, observation_noise_covariance=np.eye(4)):
        self.process_noise_variance = process_noise_variance
        self.observation_noise_covariance = observation_noise_covariance

    def predict(self, x, sigma_x):
        x_p = x
        sigma_x_p = sigma_x + self.process_noise_variance

        return x_p, sigma_x_p

    def update(self, x_p, sigma_x_p, y):
        y_pred = np.array(
            [np.cos(np.deg2rad(x_p)), np.sin(np.deg2rad(x_p)), np.cos(np.deg2rad(x_p)), np.sin(np.deg2rad(x_p))]
        )

        e = y - y_pred

        obs_jacobian = np.array(
            [[-np.sin(np.deg2rad(x_p))],
             [np.cos(np.deg2rad(x_p))],
             [-np.sin(np.deg2rad(x_p))],
             [np.cos(np.deg2rad(x_p))]]
        )

        sigma_e = self.observation_noise_covariance + obs_jacobian * sigma_x_p * np.transpose(obs_jacobian)

        kalman_gain = sigma_x_p * np.matmul(np.transpose(obs_jacobian), np.linalg.inv(sigma_e))[0]

        if np.any(np.isnan(y)):
            x_n = x_p
        else:
            x_n = x_p + np.dot(e, kalman_gain)

        sigma_x_n = (1 - np.dot(np.transpose(obs_jacobian), kalman_gain)[0]) * sigma_x_p

        return x_n, sigma_x_n

    def step(self, x, sigma_x, y):
        x_p, sigma_x_p = self.predict(x, sigma_x)
        x_n, sigma_x_n = self.update(x_p, sigma_x_p, y)

        return x_n, sigma_x_n


class StreamWeightEKF(object):
    def __init__(self, process_noise_variance=1000, observation_noise_covariance=[np.eye(2), np.eye(2)]):
        self.process_noise_variance = process_noise_variance
        self.observation_noise_covariance = observation_noise_covariance

    def predict(self, x, sigma_x):
        """
        Performs a single prediction step using a linear one-dimensional Brownian motion model.

        :param x: scalar state (azimuth in degrees) at previous time-step.
        :param sigma_x: state variance at previous time-step.
        :return: predicted state and state variance at current time step.
        """
        x_p = x
        sigma_x_p = sigma_x + self.process_noise_variance

        return x_p, sigma_x_p

    def update(self, x_p, sigma_x_p, y, w=0.5):
        """
        Performs a single update step incorporating dynamic stream weights for acoustic and visual observations.

        :param x_p: predicted state (azimuth) at current time-step.
        :param sigma_x_p: predicted state variance at current time-step.
        :param y: acoustic and visual observations as list in two-dimensional rotating vector space.
        :param w: dynamic stream weights (0: video only, 1: audio only)
        :return: estimated state and state variance.
        """

        # Compute predicted acoustic and visual observations and compute observation residuals for both modalities.
        y_pred = np.array([np.cos(np.deg2rad(x_p)), np.sin(np.deg2rad(x_p))])

        e_audio = y[0] - y_pred
        e_video = y[1] - y_pred

        # Compute observation Jacobian (which is identical for acoustic and visual observations).
        obs_jacobian = np.array([[-np.sin(np.deg2rad(x_p))], [np.cos(np.deg2rad(x_p))]])

        # Compute joint Kalman gain matrix via linear regression.
        data_matrix = np.block([
            [self.observation_noise_covariance[0] + w * obs_jacobian * sigma_x_p * np.transpose(obs_jacobian),
             (1 - w) * obs_jacobian * sigma_x_p * np.transpose(obs_jacobian)],
            [w * obs_jacobian * sigma_x_p * np.transpose(obs_jacobian),
             self.observation_noise_covariance[1] + (1 - w) * obs_jacobian * sigma_x_p * np.transpose(obs_jacobian)]
        ])

        response_matrix = np.block([[obs_jacobian * sigma_x_p], [obs_jacobian * sigma_x_p]])

        joint_kalman_gain = np.dot(np.linalg.inv(data_matrix), response_matrix)

        # Extract individual Kalman gains for both modalities.
        kalman_gain_audio = joint_kalman_gain[0:2]
        kalman_gain_video = joint_kalman_gain[2:]

        # Compute update step depending on the available observations.
        if np.any(np.isnan(y[0])) and not np.any(np.isnan(y[1])):
            # Update visual modality only of no acoustic observations are available.

            x_n = x_p + np.dot(e_video, kalman_gain_video)[0]
        elif not np.any(np.isnan(y[0])) and np.any(np.isnan(y[1])):
            # Update acoustic modality only of no visual observations are available.

            x_n = x_p + np.dot(e_audio, kalman_gain_audio)[0]
        elif not np.any(np.isnan(y[0])) and not np.any(np.isnan(y[1])):
            # Do full update if both modalities are available.

            x_n = x_p + w * np.dot(e_audio, kalman_gain_audio)[0] + (1 - w) * np.dot(e_video, kalman_gain_video)[0]
        else:
            # Skip update step if no observations are available.

            x_n = x_p

        sigma_x_n = (1 - w * np.dot(np.transpose(kalman_gain_audio), obs_jacobian)[0] -
                     (1 - w) * np.dot(np.transpose(kalman_gain_video), obs_jacobian)[0]) * sigma_x_p

        return x_n, np.maximum(sigma_x_n, 1e-9)

    def step(self, x, sigma_x, y, w=0.5):
        """
        Computes a complete recursive update step of the extended Kalman filter with dynamic stream weights.

        :param x: scalar state (azimuth in degrees) at previous time-step.
        :param sigma_x: state variance at previous time-step.
        :param y: acoustic and visual observations as list in two-dimensional rotating vector space.
        :param w: two-dimensional vector of acoustic (first element) and visual (second element) dynamic stream weights.
        :return: estimated state and state variance.
        """
        x_p, sigma_x_p = self.predict(x, sigma_x)
        x_n, sigma_x_n = self.update(x_p, sigma_x_p, y, w)

        return x_n, sigma_x_n


def load_sequence_data(data_file):
    """
    Returns ground-truth state trajectory and audiovisual observations from *.csv data file.

    :param data_file: path to *.csv data file.
    :return: state and observation trajectories.
    """
    reader = csv.reader(open(data_file, "r"))
    raw_data = list(reader)
    data_matrix = np.array(raw_data).astype("float")

    x = data_matrix[:, 4:]
    y = [data_matrix[:, 0:2], data_matrix[:, 2:4]]

    return x, y

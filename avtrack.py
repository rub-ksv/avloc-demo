import numpy as np
import filters
import cv2


def nothing(x):
    """
    Helper function for trackbars.
    """
    pass


def rvs2angle(rvs_in):
    """
    Convert from rotating vector space to azimuth angle.

    :param rvs_in: two-dimensional input in rotating vector space.
    :return: corresponding azimuth angle.
    """
    return np.rad2deg(np.arctan2(rvs_in[1], rvs_in[0]))


def main():
    cv2.namedWindow('frame')
    cv2.createTrackbar('W', 'frame', 0, 100, nothing)
    cv2.createTrackbar('Q', 'frame', 0, 100, nothing)
    cv2.createTrackbar('R_A', 'frame', 1, 100, nothing)
    cv2.createTrackbar('R_V', 'frame', 1, 100, nothing)

    cv2.setTrackbarMin('R_A', 'frame', 1)
    cv2.setTrackbarMin('R_V', 'frame', 1)

    cv2.setTrackbarPos('W', 'frame', 50)
    cv2.setTrackbarPos('Q', 'frame', 10)
    cv2.setTrackbarPos('R_A', 'frame', 10)
    cv2.setTrackbarPos('R_V', 'frame', 10)

    ekf = filters.EKF(
        process_noise_variance=cv2.getTrackbarPos('Q', 'frame'),
        observation_noise_covariance=np.diag(np.array([
            cv2.getTrackbarPos('R_A', 'frame'),
            cv2.getTrackbarPos('R_A', 'frame'),
            cv2.getTrackbarPos('R_V', 'frame'),
            cv2.getTrackbarPos('R_V', 'frame')
        ]))
    )

    swekf = filters.StreamWeightEKF(
        process_noise_variance=cv2.getTrackbarPos('Q', 'frame'),
        observation_noise_covariance=[
            cv2.getTrackbarPos('R_A', 'frame') * np.eye(2),
            cv2.getTrackbarPos('R_V', 'frame') * np.eye(2)]
    )
    x, y = filters.load_sequence_data("data.csv")

    video = cv2.VideoCapture("video.m4v")

    frame_idx = 0

    x_ekf = 0
    sigma_x_ekf = 10

    x_swekf = 0
    sigma_x_swekf = 10

    while True:
        has_frame, frame = video.read()

        if has_frame:
            y_ekf = np.hstack((y[0][frame_idx], y[1][frame_idx]))
            y_swekf = [y[0][frame_idx], y[1][frame_idx]]
            w = cv2.getTrackbarPos('W', 'frame') / 100

            process_noise_variance = cv2.getTrackbarPos('Q', 'frame')
            observation_noise_variance_audio = cv2.getTrackbarPos('R_A', 'frame') / 10
            observation_noise_variance_video = cv2.getTrackbarPos('R_V', 'frame') / 10

            ekf.process_noise_variance = process_noise_variance
            ekf.observation_noise_covariance = np.diag(np.array([
                observation_noise_variance_audio,
                observation_noise_variance_audio,
                observation_noise_variance_video,
                observation_noise_variance_video
            ]))

            swekf.process_noise_variance = process_noise_variance
            swekf.observation_noise_covariance = [
                observation_noise_variance_audio * np.eye(2),
                observation_noise_variance_video * np.eye(2)]

            x_ekf, sigma_x_ekf = ekf.step(x_ekf, sigma_x_ekf, y_ekf)
            x_swekf, sigma_x_swekf = swekf.step(x_swekf, sigma_x_swekf, y_swekf, w)

            azimuth_audio = rvs2angle(y_swekf[0])
            if not np.isnan(azimuth_audio):
                pixel_pos_audio = int((1 - (azimuth_audio + 30.5) / 61) * 640)
                cv2.line(frame, (pixel_pos_audio, 0), (pixel_pos_audio, 511), (255, 0, 0), 1)

            azimuth_video = rvs2angle(y_swekf[1])
            if not np.isnan(azimuth_video):
                pixel_pos_video = int((1 - (azimuth_video + 30.5) / 61) * 640)
                cv2.line(frame, (pixel_pos_video, 0), (pixel_pos_video, 511), (0, 255, 0), 1)

            azimuth_grid = np.arange(-60, 61, 0.01)

            x_pdf_ekf = 1/(sigma_x_ekf * np.sqrt(2 * np.pi)) * np.exp(-(azimuth_grid - x_ekf)**2 / (2 * sigma_x_ekf**2))
            x_pdf_ekf = x_pdf_ekf / np.amax(x_pdf_ekf)

            x_pdf_swekf = 1/(sigma_x_swekf * np.sqrt(2 * np.pi)) * np.exp(-(azimuth_grid - x_swekf)**2 / (2 * sigma_x_swekf**2))
            x_pdf_swekf = x_pdf_swekf / np.amax(x_pdf_swekf)

            pixel_grid = -320 + (1 - (azimuth_grid + 60) / 120) * 1280

            pixel_pdf_ekf = 485 - (x_pdf_ekf * 100)
            points_ekf = np.int32(np.stack((pixel_grid, pixel_pdf_ekf), axis=-1))
            cv2.polylines(frame, [points_ekf], True, (255, 0, 255))

            pixel_pdf_swekf = 485 - (x_pdf_swekf * 100)
            points_swekf = np.int32(np.stack((pixel_grid, pixel_pdf_swekf), axis=-1))
            cv2.polylines(frame, [points_swekf], True, (0, 255, 255))

            cv2.imshow('frame', frame)
            cv2.waitKey(33)

            frame_idx += 1

            if cv2.waitKey(33) & 0xFF == ord("q"):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

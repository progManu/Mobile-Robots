import numpy as np

from src.Utilities import wrap_angle


class Controller:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.proportional_term = 0
        self.integral_term = 0
        self.derivative_term = 0
        self.prev_error = 0

    def pid(self, target, current, dt):
        target_wrap = wrap_angle(target)
        current_wrap = wrap_angle(current)

        error = target_wrap - current_wrap

        error = np.arctan2(np.sin(error), np.cos(error))  # basically does the same as the old if stuff

        self.proportional_term = self.kp * error
        self.integral_term = self.integral_term + (self.ki * error * dt)
        self.derivative_term = self.kd * ((error - self.prev_error) / dt)

        delta = self.proportional_term + self.integral_term + self.derivative_term

        self.prev_error = error

        return delta

    def advanced_control(self, target, current, dt, cruise_velocity):
        target_wrap = wrap_angle(target)
        current_wrap = wrap_angle(current)

        error = target_wrap - current_wrap
        error = np.arctan2(np.sin(error), np.cos(error))  # basically does the same as the old if stuff

        if abs(error) > np.pi / 8:
            if np.sign(error) > 0:
                return 0, -2 * cruise_velocity
            else:
                return -2 * cruise_velocity, 0
        else:
            delta = self.pid(target, current, dt)

        return delta, -delta

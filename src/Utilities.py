import math

def wrap_angle(angle):
    two_pi = 2 * math.pi
    number_of_wraps = abs(angle)//two_pi
    if angle >= 0:
        wrap_angle = angle - (number_of_wraps * two_pi)
    else:
        wrap_angle = two_pi + angle + (number_of_wraps * two_pi)
    return wrap_angle
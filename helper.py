import numpy as np
import tw

class Helper:
    counter = 0
    stage = None

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # first
        b = np.array(b)  # mid
        c = np.array(c)  # end

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def patient_attention(self, angle_elbow, angle_wrist):
        if 30 < angle_elbow < 96:
            if self.counter == 4:
                # tw.send_message("!!! PATIENT NEED ATTENTION !!!")
                print("Patient needs attention")
                self.counter = 0
            if angle_wrist > 12:
                self.stage = "down"
            if angle_wrist < 11 and self.stage == "down":
                self.stage = "up"
                self.counter += 1
                print(self.counter)

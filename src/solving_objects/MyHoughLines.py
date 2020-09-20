import numpy as np

thresh_rho = 10
thresh_theta = 7.0 * np.pi / 180


class MyHoughLines:

    def __init__(self, line_raw):
        rho, theta = line_raw[0]
        # if theta > np.pi/2:
        #     self.theta = np.pi - theta
        #     self.rho = - rho
        # else:
        #     self.theta = theta
        #     self.rho = rho
        self.theta = min(theta, np.pi - theta)
        self.rho = rho

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        self.x1 = int(x0 + 1000 * -b)
        self.y1 = int(y0 + 1000 * a)
        self.x2 = int(x0 - 1000 * -b)
        self.y2 = int(y0 - 1000 * a)

        self.isMerged = False
        self.number_of_merged = 1

    def __str__(self):
        string = "rho : {:.1f} ".format(self.rho)
        string += "theta : {:.5f} ".format(self.theta)
        string += "x1 : {} ".format(self.x1)
        string += "x2 : {} ".format(self.x2)
        string += "y1 : {} ".format(self.y1)
        string += "y2 : {}".format(self.y2)

        return string

    def get_limits(self):
        return self.x1, self.y1, self.x2, self.y2

    def set_merged(self):
        self.isMerged = True

    def increase_count(self):
        self.number_of_merged += 1

    def recalculate_limits(self):
        a = np.cos(self.theta)
        b = np.sin(self.theta)
        x0 = a * self.rho
        y0 = b * self.rho
        self.x1 = int(x0 + 1000 * -b)
        self.y1 = int(y0 + 1000 * a)
        self.x2 = int(x0 - 1000 * -b)
        self.y2 = int(y0 - 1000 * a)

        if self.x1 < 0 and self.x2 < 0:
            self.x1, self.x2 = -self.x1, -self.x2

    def merge_w_line2(self, line2):
        self.increase_count()
        n = self.number_of_merged

        if self.rho * line2.rho > 0:
            self.rho = self.rho * (n - 1) / n + line2.rho / n
        else:
            self.rho = self.rho * (n - 1) / n - line2.rho / n
        self.theta = self.theta * (n - 1) / n + line2.theta / n

        line2.set_merged()

        self.recalculate_limits()


def are_mergeable(line1, line2):
    return abs(abs(line1.rho) - abs(line2.rho)) < thresh_rho and \
           abs(line1.theta - line2.theta) < thresh_theta


def merge_lines(lines):
    for i in range(len(lines)):
        if lines[i].isMerged:
            continue
        for j in range(i + 1, len(lines)):
            if lines[j].isMerged:
                continue
            if are_mergeable(lines[i], lines[j]):
                lines[i].merge_w_line2(lines[j])

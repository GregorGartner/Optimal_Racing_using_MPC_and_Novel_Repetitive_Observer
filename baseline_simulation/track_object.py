from math import cos, sin, fabs


class Line:
    def __init__(self, length):
        self.length = length
        self.arclength = length

    def advance(self, arclength, x, y, angle):
        return x + arclength * cos(angle), y + arclength * sin(angle), angle


class Turn:
    def __init__(self, radius, angle):
        self.radius = radius
        self.angle = angle
        self.arclength = radius * fabs(angle)

    def advance(self, arclength, x, y, angle):
        advance_angle = arclength / self.radius * fabs(self.angle) / self.angle
        advance_x = self.radius * sin(advance_angle) * fabs(self.angle) / self.angle
        advance_y = self.radius * (1 - cos(advance_angle)) * fabs(self.angle) / self.angle
        return x + advance_x * cos(angle) - advance_y * sin(angle), \
               y + advance_x * sin(angle) + advance_y * cos(angle), \
               angle + advance_angle


class Track:
    def __init__(self):
        self.parts = []

    def add_line(self, length):
        self.parts.append(Line(length))

    def add_turn(self, radius, angle):
        self.parts.append(Turn(radius, angle))

    def track_length(self):
        l = 0
        for part in self.parts:
            l += part.arclength
        return l

    def point_at_arclength(self, arclength):
        part_index = 0
        t = 0
        x = 0
        y = 0
        angle = 0

        while True:
            part = self.parts[part_index]
            if arclength - t > part.arclength:
                t += part.arclength
                x, y, angle = part.advance(part.arclength, x, y, angle)
            else:
                x, y, angle = part.advance(arclength - t, x, y, angle)
                return x, y

            part_index += 1
            if part_index >= len(self.parts):
                part_index = 0
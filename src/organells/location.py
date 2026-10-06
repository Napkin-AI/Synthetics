import numpy as np
import math

class Location:
    def __init__(self):
        self.numberPoints = 0
        self.Points = []
        self.PointsWithOffset = []

        self.centerPoint = [0,0]
        self.angle = 0
        
    def __copy__(self):
        new_loc = Location()

        new_loc.numberPoints = self.numberPoints
        new_loc.Points = self.Points.copy()
        new_loc.PointsWithOffset = self.PointsWithOffset.copy()

        new_loc.centerPoint = [self.centerPoint[0], self.centerPoint[1]]
        new_loc.angle = self.angle
        
        return new_loc
        
    def copy(self):
        return self.__copy__()
        
        
    def setAngle(self, change_angle):
        change_angle = math.radians(change_angle)

        tPoints = []
        for point in self.Points:
            x = int(round(point[0] * math.cos(change_angle) - point[1] * math.sin(change_angle)))
            y = int(round(point[0] * math.sin(change_angle) + point[1] * math.cos(change_angle)))
            tPoints.append([x,y])

        self.Points = tPoints
        self.ChangePositionPoints()

    def setRandomAngle(self, min_angle = 0, max_angle = 90, is_singned_change = True):

        if np.random.random() < 0.5 and is_singned_change:
            sign = -1
        else:
            sign = 1

        new_angle = (self.angle + np.random.randint(min_angle, max_angle+1) * sign) %360
        change_angle = (new_angle - self.angle)
        self.angle = new_angle

        self.setAngle(change_angle)

    def ChangePositionPoints(self):
        self.PointsWithOffset = []

        for point in self.Points:
            self.PointsWithOffset.append([self.centerPoint[0]+point[0], self.centerPoint[1]+point[1]])

    def NewPosition(self, x, y):
        self.centerPoint[0] = x
        self.centerPoint[1] = y

        self.ChangePositionPoints()
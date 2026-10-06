from src.container.spline import *

class Pen:
    def __init__(self, color = (0,0,0), sizePen = 1):
        self.color = color
        self.sizePen = sizePen

    def copy(self):
        new_pen = Pen()
        new_pen.color = self.color
        new_pen.sizePen = self.sizePen
        return new_pen

class Brush:
    def __init__(self, brush = (0,0,0), typeFull = "full"):
        self.brush = brush
        self.type = typeFull

    def copy(self):
        new_brush = Brush()
        new_brush.brush = self.brush
        new_brush.type = self.type
        return new_brush

    def FullBrush(self, image, points):
        if self.type == "full":
            fill_full_spline(image, points, self.brush)
        elif self.type == "texture":
            fill_texture_spline(image, points, self.brush)
        else:
            raise Exception("There is no type", self.type, "using only 'full' or 'texture'")

    def FullBrushEllipse(self, image, center, radius, angle = 0):
        if self.type == "full":
            fill_full_ellipse(image, center, radius, self.brush, angle)
        elif self.type == "texture":
            fill_texture_ellipse(image, center, radius, self.brush, angle)
        else:
            raise Exception("There is no type", self.type, "using only 'full' or 'texture'")

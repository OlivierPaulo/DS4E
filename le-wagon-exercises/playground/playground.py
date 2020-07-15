# TODO: it's a playground, let's write some code (no unit tests to run here)
import math

def circle_math(radius):
    perimeter = round(radius * 2 * math.pi,1)
    area = round(radius**2 * math.pi,1)
    return [perimeter,area]

print ("Please choose the radius of a circle : ")
radius = int(input())

values = circle_math(radius)
print(f"Radius={radius} => Perimeter={values[0]}, Area={values[1]}")



# TODO: it's a playground, let's write some code (no unit tests to run here)
import math


print ("Please choose the radius of a circle : ")
radius = int(input())
perimeter = round(radius * 2 * math.pi,1)

print(f"Radius of the circle is : {radius}")
print(f"Perimeter of the circle is : {perimeter}")

area = round(radius**2 * math.pi,1)
print(f"Area of the circle is : {area}")


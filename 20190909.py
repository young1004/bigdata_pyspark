def my_range(begin, end, step=1) :
  x = begin
  while x < end :
    print(x)
    x += step

def sum(list, f = lambda x: x) :
  result = 0
  for x in list :
    result += f(x)
  return result

def sq(x) :
  return x * x

def cube(x) :
  return x ** 3

### 2019/09/11 ###

class Circle:
  def __init__(self, r = 1):
    self.radius = r
  def getArea(self):
    return self.radius * self.radius * math.pi
  def setRadius(self, new_r):
    self.radius = new_r
  def __add__(self, other):
    self.radius += other.radius
  def __lt__(self, other):
    return self.radius < other.radius
  def __str__(self):
    return "Circle with r = " + str(self.radius)

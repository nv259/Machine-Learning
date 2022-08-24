import numpy as np

# Tạo mảng
a = np.array([1,2,3])
print("1D array")
print(a)

# Xác định kiểu của mảng
print("Type: ", type(a))

# Tạo mảng 2 chiều (ma trận)
b = np.array([[1,2,3],
              [4,5,6]])
print("2D array")
print(b)

# Tạo mảng 3 chiều (tensor)
c = np.array([[[1,2,3], [4,5,6]], 
              [[7,8,9], [10,11,12]]])
print("3D array")
print(c)

# Mảng 2 chiều chứa 1 dòng, nhưng không phải mảng 1 chiều
d = np.array([[1,2,3]])

# In ra kích thước các mảng
print("Shape of a: ", a.shape)
print("Shape of a: ", b.shape)
print("Shape of a: ", c.shape)
print("Shape of a: ", d.shape)

# Các ma trận khác
print("Matrix of ones")
print(np.ones((2,2)))
print("Matrix of zeros")
print(np.zeros((2,2)))
print("Identity matrix")
print(np.eye(3))
print("Random matrix")
print(np.random.rand(3,2))
# Các toán tử cơ bản
a = np.array([[1,0,2],
              [2,3,0]])
b = np.array([[2,1,-1],
              [2,1,2]])

print("a+b: ", a+b)
print("a-b: ", a-b)
print("a*b: ", a*b)
print("a/b: ", a/b)

# Broadcasting
a = np.array([[1,0,2],
              [2,3,0]])
b = np.array([[1,2,1]])
c = np.array([[1],
              [2]])
print("Broadcasting")
print(a-1)
print(a-b)
print(a-c)
# Chuyển vị ma trận
b = np.array([[2,1,-1],
              [2,1,2]])
print("b: ", b)
print("b transpose: ", b.T)

# Nhân ma trận
mul = np.matmul(a, b.T)
print("a.bT: ", mul)

# Truy cập mảng
a = np.array([[1,0,2],
              [2,3,0]])
print(a[1,0])
print(a[1,:])
print(a[:,1])
print(a[:2, :2])

# Masking
a = np.array([[1,0,2],
              [2,3,0]])
mask = a>=2
print("Elements that greater than or equal 2")
print(a[mask])

# Thay đổi phần tử trong mảng
a = np.array([[1,0,2],
              [2,3,0]])
print("Changed (0,1)")
a[0,1] = 5
print(a)
print("Changed column 0")
a[:,0] = np.zeros((2,))
print(a)

print("Change with masking")
a = np.array([[1,0,2],
              [2,3,0]])
a[mask] = -1
print(a)

# Thay đổi kích thước mảng
a = np.array([[1,0,2],
              [2,3,0]])
print("Reshape to 3x2")
print(a.reshape((3,2)))
print("Reshape to one row (still 2D)")
print(a.reshape((1,-1)))

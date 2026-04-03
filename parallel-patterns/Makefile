TARGET = stencil-3d-tiled
SRC = stencil-3d-tiled.cu

all:
	nvcc -arch=sm_75 $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

profile: all
	ncu --set full ./$(TARGET)

clean:
	rm -f $(TARGET)
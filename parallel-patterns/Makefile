TARGET = histogram
SRC = histogram.cu

all:
	nvcc -arch=sm_75 $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

profile: all
	ncu --set full ./$(TARGET)

clean:
	rm -f $(TARGET)
TARGET = brent-kung-scan-exclusive
SRC = brent-kung-scan-exclusive.cu

all:
	nvcc -arch=sm_75 $(SRC) -o $(TARGET)

run: all
	./$(TARGET)

profile: all
	ncu --set full ./$(TARGET)

clean:
	rm -f $(TARGET)
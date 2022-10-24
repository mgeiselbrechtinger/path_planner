BUILD_DIR=build
APP=path-planner
SRC=main

all: $(BUILD_DIR)/$(APP)

$(BUILD_DIR)/$(APP): $(BUILD_DIR)/$(SRC).o
	gcc $< -o $@ -lstdc++ -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui

$(BUILD_DIR)/$(SRC).o: $(SRC).cpp
	gcc -c -Wall -I/usr/include/opencv4 -o $@ $<

clean:
	rm $(BUILD_DIR)/*

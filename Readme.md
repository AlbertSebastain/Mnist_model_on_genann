# Train a Handwriting Recognition Model Based on MNIST Dataset

This is a simple handwriting number recognition model implemented in pure C. The model is trained on the MNIST dataset using the open-source framework [genann](https://github.com/codeplea/genann?tab=readme-ov-file)

## Usage

windows
```bash
mkdir build
cd ./build
cmake -G "MinGW Makefiles" ..
mingw32-make
./mnist_in_c.exe
```
linux
```bash
mkdir build
cd ./build
cmake ..
make
./mnist_in_c.out
```

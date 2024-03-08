# Importing required libraries
import sys
import numpy as np
import cv2

# Combining multiple images both horizontally and vertically,
# resizing the final image and 
# displaying it in a single window for Task 1
def plotImage2x2(originalImage, c1, c2, c3):
	# Convert 2-channel grayscale images to 3-channel color images
	cs1 = cv2.cvtColor(c1, cv2.COLOR_GRAY2BGR)
	cs2 = cv2.cvtColor(c2, cv2.COLOR_GRAY2BGR)
	cs3 = cv2.cvtColor(c3, cv2.COLOR_GRAY2BGR)

	# Horizontally stacked two images together and the two horizontally stacked image groups are then vertically stacked
	imgGroup1 = np.hstack((originalImage, cs1))
	imgGroup2 = np.hstack((cs2, cs3))
	imgGroup3 = np.vstack((imgGroup1, imgGroup2))

	# Final image array to have dimensions 1280 x 720 pixels
	finalConvertedImages = cv2.resize(imgGroup3, (1280, 720))

	# Creating a resizable window and display the final image
	cv2.namedWindow('Task 1', cv2.WINDOW_FULLSCREEN)
	cv2.imshow('Task 1', finalConvertedImages)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Combining multiple images both horizontally and vertically,
# resizing the final image and 
# displaying it in a single window for Task 2
def originalImageToColorSpace(inputOption, inputImage):
	# Converting image to specified color space
	# Splitting the specified color space channels of image
	# plotting the 4 images
	if inputOption == "-XYZ":
		convertedImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2XYZ)
		c1, c2, c3 = cv2.split(convertedImage)

		plotImage2x2(inputImage, c1, c2, c3)
	elif inputOption == '-LAB':
		convertedImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2LAB)
		c1, c2, c3 = cv2.split(convertedImage)

		plotImage2x2(inputImage, c1, c2, c3)
	elif inputOption == '-YCRCB':
		convertedImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2YCrCb)
		c1, c2, c3 = cv2.split(convertedImage)

		plotImage2x2(inputImage, c1, c2, c3)
	elif inputOption == "-HSB":
		convertedImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2HSV)
		c1, c2, c3 = cv2.split(convertedImage)

		plotImage2x2(inputImage, c1, c2, c3)

def greenImageToScenicImage(greenImage, scenicImage):
	# Making equal size for both the images
	resizedGreenImage = cv2.resize(greenImage, (scenicImage.shape[1], scenicImage.shape[0]))

	# Define upper and lower range in HSV
	greenUpper = np.array([75, 255, 255])
	greenLower = np.array([40, 40, 40])

	# Create a binary mask to identify green regions based on specified range
	hsvGreenImage = cv2.cvtColor(resizedGreenImage, cv2.COLOR_BGR2HSV)
	binaryMask = cv2.inRange(hsvGreenImage, greenLower, greenUpper)

	# Create a new image that isolates the green color regions from the green image, based on the binary mask
	# New image include green background and makes the other regions black
	greenColorImage = np.where(binaryMask[..., None], resizedGreenImage, 0)

	# Create the final image by removing green from the green image and get a black background image
	noGreenImage = resizedGreenImage - greenColorImage

	# Replace black background with white for white background
	whiteImage = np.where(noGreenImage == 0, 255, noGreenImage)

	# Replace black background with scenicImage
	combinedImage = np.where(noGreenImage == 0, scenicImage, noGreenImage)

	# Horizontally stacked two images together and the two horizontally stacked image groups are then vertically stacked
	imgGroup1 = np.hstack((resizedGreenImage, whiteImage))
	imgGroup2 = np.hstack((scenicImage, combinedImage))
	imgGroup3 = np.vstack((imgGroup1, imgGroup2))
	
	# Final image array to have dimensions 1280 x 720 pixels
	finalConvertedImages = cv2.resize(imgGroup3, (1280, 720))

	# Creating a resizable window and display the final image
	cv2.namedWindow('Task 2', cv2.WINDOW_FULLSCREEN)
	cv2.imshow('Task 2', finalConvertedImages)

	cv2.waitKey(0)
	cv2.destroyAllWindows()

# Python driver code
if __name__ == '__main__':
	# Accepting input arguments from command line
	inputArgs = sys.argv

	# Color spaces options array
	colorSpaceOptions = ['-XYZ', '-LAB', '-YCRCB', '-HSB']

	# Input paramters checking for correctness
	if len(inputArgs) <= 2 or (inputArgs[1][0] == '-' and inputArgs[1].upper() not in colorSpaceOptions):
		print("Please enter valid arguments. Follow the instructions specified below:")
		print("Accepted arguments are:")
		print('#Task 1 - "ChromaKey -XYZ|-Lab|-YCrCb|-HSB imagefile"')
		print('#Task 2 - "ChromaKey scenicImageFile greenScreenImagefile"')
		sys.exit(0)
	
	# Destructuring the input arguments
	input1 = inputArgs[1]
	input2 = inputArgs[2]
	
	# Checking the input arguments for validity to separate for both tasks
	inputOption = input1.upper()
	inputImage = cv2.imread(input2)

	if (inputOption in colorSpaceOptions):
		# Calling the task 1 function
		originalImageToColorSpace(inputOption, inputImage)
	else:
		# Reading input images
		scenicImage = cv2.imread(input1)
		greenImage = cv2.imread(input2)

		# Input paramters checking for correctness
		if scenicImage is None or greenImage is None or scenicImage.size == 0 or greenImage.size == 0:
			print("Please enter valid arguments. Follow the instructions specified below:")
			print("Accepted arguments are:")
			print('#Task 1 - "ChromaKey -XYZ|-Lab|-YCrCb|-HSB imagefile"')
			print('#Task 2 - "ChromaKey scenicImageFile greenScreenImagefile"')
			sys.exit(0)
		else:
			# Calling the task 2 function
			greenImageToScenicImage(greenImage, scenicImage)
######################
## Essential libraries
######################
import cv2
import numpy as np
import os
import math
import copy


codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Videos'))
generated_folder_path = os.path.abspath(os.path.join('..', 'Generated'))




def kernel_psf(angle, d, size=20):
    kernel = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    size2 = size // 2                                              # Division(floor)
    A[:,2] = (size2, size2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kernel = cv2.warpAffine(kernel, A, (size, size), flags=cv2.INTER_CUBIC)   # image to specific matrix conversion
    return kernel

#wiener filter implementaion
def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    copy_img = np.copy(img)
    copy_img = np.fft.fft2(copy_img)            #  2D fast fourier transform 
    kernel = np.fft.fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)     # wiener formula implementation
    copy_img = copy_img * kernel                             # conversion blurred to deblurred
    copy_img = np.abs(np.fft.ifft2(copy_img))   # 2D inverse fourier transform
    return copy_img

def process(ip_image):
    a=2.2                                                          # contrast
    ang=np.deg2rad(90)                                             # angle psf
    d=20                                                         # distance psf
    
    b, g, r = cv2.split(ip_image)

    # normalization of split images 
    img_b = np.float32(b)/255.0
    img_g = np.float32(g)/255.0
    img_r = np.float32(r)/255.0
    #psf calculation 

    psf = kernel_psf(ang, d)
    #wiener for all split images
    filtered_img_b = wiener_filter(img_b, psf, K = 0.0060)          # small value of k that is snr as if 0 filter will be inverse filter 
    filtered_img_g = wiener_filter(img_g, psf, K = 0.0060)
    filtered_img_r = wiener_filter(img_r, psf, K = 0.0060)
    #merge to form colored image
    filtered_img=cv2.merge((filtered_img_b,filtered_img_g,filtered_img_r))
    #converting float to unit 
    filtered_img=np.clip(filtered_img*255,0,255)   # clipping values between 0 and 255
    filtered_img=np.uint8(filtered_img)
    #changing contrast of the image
    filtered_img=cv2.convertScaleAbs(filtered_img,alpha=a)
    #removing gibbs phenomena or rings from the image
    filtered_img = cv2.fastNlMeansDenoisingColored(filtered_img, None, 10, 10, 7, 15) 
    filtered_img = cv2.fastNlMeansDenoisingColored(filtered_img, None, 10, 10, 7, 15) # removing left over rings in post processing again     
   
    # using unblurred image to get angle and id of aruco
    return filtered_img


    

def main(val):
    ################################################################
    ## variable declarations
    ################################################################
    i = 1
    ## reading in video 
    cap = cv2.VideoCapture(images_folder_path+"/"+"video name.mp4")
    ## getting the frames per second value of input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    ## getting the frame sequence
    frame_seq = int(val)*fps
    ## setting the video counter to frame sequence
    cap.set(1,frame_seq)
    ## reading in the frame
    ret, frame = cap.read()
    ## verifying frame has content
    print(frame.shape)
    ## display to see if the frame is correct
    cv2.imshow("window", frame)
    cv2.waitKey(0);
    ## calling the algorithm function
    op_image = process(frame)


    

############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main(input("time value in seconds:"))



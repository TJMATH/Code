import numpy as np

bin_n = 16*16 # Number of bins

def hog(img):
    x_pixel, y_pixel = 194, 259
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)
    bins = np.int32(bin_n*ang/(2*np.pi)) # quantizing binvalues in (0,...,16)
    bin_cells = (bins[:x_pixel/2, :y_pixel/2],
                 bins[x_pixel/2:, :y_pixel/2],
                 bins[:x_pixel/2, y_pixel/2:],
                 bins[x_pixel/2:, y_pixel/2:])
    mag_cells = (mag[:x_pixel/2, :y_pixel/2],
                 mag[x_pixel/2:, :y_pixel/2],
                 mag[:x_pixel/2, y_pixel/2:],
                 mag[x_pixel/2:, y_pixel/2:])
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)      # hist is a 64 bit vector
    # print hist.shape
    # print type(hist)
    return hist

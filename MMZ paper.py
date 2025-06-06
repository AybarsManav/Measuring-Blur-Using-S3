import cv2
import numpy as np
import matplotlib.pyplot as plt

class BlurDetector(object):
    def __init__(self):
        self.dct_threshold = 8.0
        self.max_hist = 0.1
        self.hist_weight = np.array([
            8, 7, 6, 5, 4, 3, 2, 1,
            7, 8, 7, 6, 5, 4, 3, 2,
            6, 7, 8, 7, 6, 5, 4, 3,
            5, 6, 7, 8, 7, 6, 5, 4,
            4, 5, 6, 7, 8, 7, 6, 5,
            3, 4, 5, 6, 7, 8, 7, 6,
            2, 3, 4, 5, 6, 7, 8, 7,
            1, 2, 3, 4, 5, 6, 7, 8
        ]).reshape(8, 8)
        self.weight_total = 344.0

    def check_image_size(self, image, block_size=32):
        height, width = image.shape[:2]
        pad_y = (block_size - height % block_size) % block_size
        pad_x = (block_size - width % block_size) % block_size
        padded = False
        if pad_y > 0 or pad_x > 0:
            padded = True
            image = cv2.copyMakeBorder(image, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)
        return padded, image

    def get_blurness(self, image):
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        height, width = image.shape
        block_size = 32
        step = 8
        #seg_v = height // 32 

        seg_v = (height-block_size)//step + 1
        seg_h = (width-block_size)//step + 1
    
        print(seg_h)
        blur_grid = np.zeros((seg_v, seg_h))

        for v in range(seg_v):
            for h in range(seg_h):
                # Extract 32x32 segment
                start_y = v * step
                start_x = h * step
                segment = image[start_y:start_y+block_size, start_x:start_x+block_size]
                hist_seg = np.zeros((8, 8), dtype=int)
                
                # Process each 8x8 block in the segment
                for i in range(4):
                    for j in range(4):
                        block = segment[i*8:(i+1)*8, j*8:(j+1)*8]
                        block = np.float32(block)
                        dct_block = cv2.dct(block)
                        mask = (np.abs(dct_block) > self.dct_threshold)
                        hist_seg += mask.astype(int)
                
                # Calculate segment blur score
                condition = hist_seg < (self.max_hist * hist_seg[0, 0])
                weighted_blur = np.multiply(condition, self.hist_weight)
                seg_score = weighted_blur.sum() / self.weight_total
                blur_grid[v, h] = seg_score
        
        return blur_grid

if __name__ == "__main__":
    bd = BlurDetector()
    image = cv2.imread('running.png')
    if image is None:
        print('Image file does not exist!')
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, padded_img = bd.check_image_size(gray, block_size=32)
        
        # Get 8x8 blur grid and global average
        blur_grid = bd.get_blurness(padded_img)
        sharpness_grid = 1 - blur_grid

        global_sharpness = np.mean(sharpness_grid)
        #global_blur = 
        print("Global Blurness: {:.4f}".format(global_sharpness))
        

        plt.figure(figsize=(10, 10))
        plt.imshow((sharpness_grid * 255).astype(np.uint8) ,cmap='gray')

        plt.show()      
        cv2.imwrite('running_mmz.jpeg', (sharpness_grid * 255).astype(np.uint8))

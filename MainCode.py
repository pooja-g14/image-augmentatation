import random
import cv2
import numpy as np
import glob

def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list=[]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(np.random.randint(7,10)): ## Dimensionality of the shadow polygon
            vertex.append(( imshape[1]*np.random.uniform(),imshape[0]//3+imshape[0]*np.random.uniform()))
            vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices
            vertices_list.append(vertices)

    return vertices_list ## List of shadow vertices


def get_corners(bboxes):

    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape N X 4 where N is the
        number of bounding boxes and the bounding boxes are represented in the
        format x1 y1 x2 y2

    returns
    -------

    numpy.ndarray
        Numpy array of shape N x 8 containing N bounding boxes each described by their
        corner co-ordinates x1 y1 x2 y2 x3 y3 x4 y4

    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)

    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)

    x2 = x1 + width
    y2 = y1

    x3 = x1
    y3 = y1 + height

    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)

    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

    return corners


def rotate_box(image, corners, angle):

    w,h = image.shape[1], image.shape[0]
    cx, cy = w//2, h//2

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T

    calculated = calculated.reshape(-1,8)

    x_ = calculated[:,[0,2,4,6]]
    y_ = calculated[:,[1,3,5,7]]

    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)

    final = np.hstack((xmin, ymin, xmax, ymax, calculated[:,8:]))

    return final


def add_rotation(image, angle_i=None):

    if angle_i is None:
        angle = random.randint(-4, 4)
    else:
        angle = angle_i

    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    # image = cv2.resize(image, (w,h))

    return image, angle


def motion_kernel(angle, d, sz=65):
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern


def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern


def apply_filter(psf, img):

    img = np.float32(img)/255.0

    psf /= psf.sum()
    psf_pad = np.zeros_like(img)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf

    psf_dft = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT)
    img_dft = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

    res_dft = cv2.mulSpectrums(img_dft, psf_dft, 0)
    op_img = cv2.idft(res_dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )

    op_img = np.roll(op_img, -kh//2, 0)
    op_img = np.roll(op_img, -kw//2, 1)

    op_img = op_img * 255

    return op_img


def get_char_bboxs(label_txt, font, xy_loc, char_enum, skip_classes, offset=None):

    bboxs = []
    # get bbox of the upper label
    for i, char in enumerate(label_txt):

        if char in skip_classes:
            continue

        bottom_1 = font.getsize(label_txt[i])[1]
        right, bottom_2 = font.getsize(label_txt[:i+1])
        bottom = bottom_1 if bottom_1 < bottom_2 else bottom_2
        width, height = font.getmask(char).size

        if offset is None:
            right += xy_loc[0]
            bottom += xy_loc[1]
        else:
            right += (xy_loc[0] + offset[0])
            bottom += (xy_loc[1] + offset[1])

        top = bottom - height
        left = right - width

        class_idx = char_enum.index(char)

        bboxs.append((class_idx, left, top, right, bottom))

    return bboxs

class MakeRealistic:

    def __init__(self, label):

        self.label = label
        #self.bboxs = bboxs


    def change_brightness(self):
        #Change brightness and contrast
        alpha = random.uniform(1.5, 2.5)
        beta  = random.randint(-60, -20)
        self.label = cv2.convertScaleAbs(self.label, alpha=alpha, beta=beta)


    def add_gaussian_noise(self, noise_sigma=700, scale_noise=0.4):

        image_temp = self.label.copy()

        if len(self.label.shape) == 2:
            m = noise_sigma
            s = noise_sigma
        else:
            m = (noise_sigma,noise_sigma,noise_sigma)
            s = (noise_sigma,noise_sigma,noise_sigma)

        cv2.randn(image_temp,m,s)
        self.label = cv2.addWeighted(self.label, (1-scale_noise), image_temp, scale_noise, 0)


    def add_speckle_noise(self, scale=0.2):

        # row,col,ch = image.shape
        # gauss = np.random.randn(row,col,ch)
        # gauss = gauss.reshape(row,col,ch)
        # noisy_image = image + image * gauss * scale

        # return noisy_image.astype('uint8')

        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        prob = scale
        if len(self.label.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = self.label.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        # probs = np.random.random(self.label.shape[:2])
        probs = np.random.normal(loc=1.0, scale=0.2, size=self.label.shape[:2])
        self.label[probs < (prob / 2)] = black
        # self.label[probs > 1 - (prob / 2)] = white

    def rotate_label(self, angle):

        self.label, angle = add_rotation(self.label, angle_i=angle)
        #corners = get_corners(self.bboxs[:,1:])
        #rotated_bbox = rotate_box(self.label, corners, angle)
        #final_bbox = np.insert(rotated_bbox, 0, self.bboxs[:,0], axis=1)
        #self.bboxs = final_bbox


    def add_shadow(self, full_shadow=True, no_of_shadows=1):
        image_RGBA = cv2.cvtColor(self.label, cv2.COLOR_RGB2RGBA) ## Conversion to HLS

        # mask = np.zeros_like(image_RGBA)
        # imshape = self.label.shape
        # vertices_list= generate_shadow_coordinates(imshape, no_of_shadows) #3 getting list of shadow vertices

        # for vertices in vertices_list:
        #     mask = cv2.fillPoly(mask, vertices, (255, 255, 255)) ## adding all shadow polygons on empty mask, single 255 denotes only red channel

        # mask = cv2.bitwise_not(mask)


        # image_RGBA = cv2.addWeighted(image_RGBA, 0.75, mask, 0.25, -0.5)

        if full_shadow:

            alpha = random.uniform(0.25, 0.35)


            mask_full = np.ones_like(image_RGBA)
            image_RGBA = cv2.addWeighted(image_RGBA, alpha, mask_full, 1-alpha, -0.0)

        self.label = cv2.cvtColor(image_RGBA, cv2.COLOR_RGBA2RGB)
        # cv2.imshow("mask", dst)
        # cv2.waitKey(0)


    def add_motion_blur(self, len, theta):

        img = cv2.cvtColor(self.label, cv2.COLOR_RGB2GRAY)
        img = img[0:(img.shape[0] & -2), 0:(img.shape[1] & -2)]

        psf = motion_kernel(theta, len)
        mb_img = apply_filter(psf, img)

        psf_defocus = defocus_kernel(len//3)
        self.label = apply_filter(psf_defocus, mb_img)

    def dilation(self, kernel_size):
        # dilation_size = kernel_size
        # dilation_shape = cv2.MORPH_RECT
        # element = cv2.getStructuringElement(dilation_shape, (2 * dilation_size + 1, 2 * dilation_size + 1),
        #                                 (dilation_size, dilation_size))
        # self.label = cv2.erode(self.label, element)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        self.label = cv2.morphologyEx(self.label, cv2.MORPH_CLOSE, kernel)

    def run(self):

        self.change_brightness()
        self.add_shadow()
        self.dilation(kernel_size=random.randint(2,4))
        self.add_motion_blur(len=random.randint(3, 6), theta=random.randint(1, 3))
        self.add_speckle_noise(scale=0.1)
        self.add_gaussian_noise(noise_sigma=random.randint(10, 30), scale_noise=random.uniform(0.1, 0.3))
        self.rotate_label(angle=270)

        return self.label

images = [cv2.imread(file) for file in glob.glob("/home/ubuntu/data/dataset/Image_Augmentation/Train_Images/*/RCImage0001.png")]
for i in range (0, 1 + len(images)):
    a = MakeRealistic(images[i])
    modified = a.run()
    cv2.imwrite("/home/ubuntu/data/dataset/Image_Augmentation/Augmented_Images/Augmented{0}.png".format(i), modified)
    #modified = modified/255
    #cv2.imshow('mo',modified)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

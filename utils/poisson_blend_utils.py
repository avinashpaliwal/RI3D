import torch
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as linalg
from scipy.optimize import curve_fit
import cv2
import yaml
# from kornia.morphology import erosion, dilation, opening, closing
# import matplotlib.pyplot as plt

# from bilateral_filtering import sparse_bilateral_filtering
# from depth_layering import get_depth_bins

def piecewise_func(x, y, mask):
    # print(x.shape, y.shape, mask.shape)
    # print((x*mask).nonzero().shape, (y*mask).nonzero().shape)
    # print(x.max(), x.min(), y.max(), y.min(), mask.max(), mask.min())
    # Split data at the midpoint
    split_point_x = np.percentile(x[mask.nonzero()], 95)
    # split_point_x0 = min(x[mask.nonzero()].min(), np.percentile(x[mask == 0], 5))
    split_point_x0 = x[mask.nonzero()].min()
    # split_point_x = x[mask.nonzero()].max()
    print(split_point_x, split_point_x0, x.max(), x.min(), x[mask.nonzero()].min(), np.percentile(x[mask == 0], 5))

    # Define the higher-order function (cubic in this case)
    def poly_func(x, a, b, c, d, e):
        return a + b * x + c * x**2 + d * x**3 #+ e * x**4 #+ f * x**5
    
    # Fit the higher-order function to the first segment
    params1, _ = curve_fit(poly_func, x[mask.nonzero()], y[mask.nonzero()])
    
    split_point_y = poly_func(split_point_x, *params1)

    # Define the linear function
    def linear_func(x, a, b):
        # return m*x + c
        # return a*(x-split_point_x)**2 + b*(x-split_point_x) + split_point_y
        return b*(x-split_point_x) + split_point_y
    

    split_point_y0 = poly_func(split_point_x0, *params1)
    # Define the linear function
    def linear_func0(x, a, b, c):
        # return m*x + c
        # return   b*x + c
        return c*(x-split_point_x0)**3 + a*(x-split_point_x0)**2 + b*(x-split_point_x0) + split_point_y0
        # return b*(x-split_point_x0) + split_point_y0


    # Fit the linear function to the second segment
    params2, _ = curve_fit(linear_func, x[(1-mask).nonzero()], y[(1-mask).nonzero()])


    x_test = np.arange(0, 1, 0.01)
    x_test = x_test * (x.max() - x.min()) + x.min()

    if round(split_point_x0, 3) > round(x.min(), 3):
        params3, _ = curve_fit(linear_func0, x[(x < split_point_x0).nonzero()], y[(x < split_point_x0).nonzero()])

        y_test = np.piecewise(x_test, [x_test < x[mask.nonzero()].min(), np.logical_and(x_test >= x[mask.nonzero()].min(), x_test < split_point_x), x_test >= split_point_x], 
                            [lambda x: linear_func0(x, *params3), lambda x: poly_func(x, *params1), lambda x: linear_func(x, *params2)])
    else:
        y_test = np.piecewise(x_test, [x_test < split_point_x, x_test >= split_point_x], 
                            [lambda x: poly_func(x, *params1), lambda x: linear_func(x, *params2)])
    
    # plt.plot(x_test, y_test)
    # plt.figure()

    if round(split_point_x0, 3) > round(x.min(), 3):

        return np.piecewise(x, [x < x[mask.nonzero()].min(), np.logical_and(x >= x[mask.nonzero()].min(), x < split_point_x), x >= split_point_x], 
                            [lambda x: linear_func0(x, *params3), lambda x: poly_func(x, *params1), lambda x: linear_func(x, *params2)])
    else:
        return np.piecewise(x, [x < split_point_x, x >= split_point_x], 
                            [lambda x: poly_func(x, *params1), lambda x: linear_func(x, *params2)])

def preprocess(image_data):
    # extract image data
    source = image_data['source']
    mask = image_data['mask']
    target = image_data['target']

    # get image shape and offset
    Hs,Ws,_ = source.shape
    Ht,Wt,_ = target.shape
    Ho, Wo = image_data['dims']

    # adjust source and mask if offset is negative.
    # if mask is rolled eg. from the top it rolls
    # to the bottom, crop the rolled portion
    if(Ho < 0):
        mask = np.roll(mask, Ho, axis=0)
        source = np.roll(source, Ho, axis=0)
        mask[Hs+Ho:,:,:] = 0 # added because Ho < 0
        source[Hs+Ho:,:,:] = 0
        Ho = 0
    if(Wo < 0):
        mask = np.roll(mask, Wo, axis=1)
        source = np.roll(source, Wo, axis=1)
        mask[:,Ws+Wo:,:] = 0
        source[:,Ws+Wo:,:] = 0
        Wo = 0

    # mask region on target
    H_min = Ho
    H_max = min(Ho + Hs, Ht)
    W_min = Wo
    W_max = min(Wo + Ws, Wt)

    # crop source and mask if they lie outside the bounds of the target
    source = source[0:min(Hs, Ht-Ho),0:min(Ws, Wt-Wo),:]
    mask = mask[0:min(Hs, Ht-Ho),0:min(Ws, Wt-Wo),:]

    return {'source':source, 'mask': mask, 'target': target, 'dims':[H_min,H_max,W_min,W_max]}

def get_subimg(image, dims):
    return image[dims[0]:dims[1], dims[2]:dims[3]]

def poisson_blending(image, GRAD_MIX, fg_mask=None):
    # comparison function
    def _compare(val1, val2):
        if(abs(val1) > abs(val2)):
            return val1
        else:
            return val2

    # membrane (region where Poisson blending is performed)
    mask = image['mask']
    Hs,Ws = mask.shape
    num_pxls = Hs * Ws

    # source and target image
    source = image['source'].flatten(order='C')
    target_subimg = get_subimg(image['target'], image['dims']).flatten(order='C')

    # initialise the mask, guidance vector field and laplacian
    mask = mask.flatten(order='C')
    guidance_field = np.empty_like(mask)
    laplacian = sps.lil_matrix((num_pxls, num_pxls), dtype='float64')

    for i in range(num_pxls):
        # construct the sparse laplacian block matrix
        # and guidance field for the membrane
        # if(mask[i] > 0.99):
        if True:

            laplacian[i, i] = 4

            # construct laplacian, and compute source and target gradient in mask
            if(i - Ws > 0):
                laplacian[i, i-Ws] = -1
                Np_up_s = source[i] - source[i-Ws]
            else:
                Np_up_s = source[i]

            if(i % Ws != 0):
                laplacian[i, i-1] = -1
                Np_left_s = source[i] - source[i-1]
            else:
                Np_left_s = source[i]

            if(i + Ws < num_pxls):
                    laplacian[i, i+Ws] = -1
                    Np_down_s = source[i] - source[i+Ws]
            else:
                Np_down_s = source[i]

            if(i % Ws != Ws-1):
                laplacian[i, i+1] = -1
                Np_right_s = source[i] - source[i+1]
            else:
                Np_right_s = source[i]

            # choose stronger gradient
            if(GRAD_MIX is False):
                Np_up_t = 0
                Np_left_t = 0
                Np_down_t = 0
                Np_right_t = 0

            guidance_field[i] = (_compare(Np_up_s, Np_up_t) + _compare(Np_left_s, Np_left_t) +
                                _compare(Np_down_s, Np_down_t) + _compare(Np_right_s, Np_right_t))

        else:
            # if point lies outside membrane, copy target function
            laplacian[i, i] = 1 * 0.1
            guidance_field[i] = target_subimg[i] * 0.1

        if mask[i] < 0.99:
            # if point lies outside membrane, copy target function
            laplacian[i, i] += 1 * 1
            guidance_field[i] += target_subimg[i] * 1

        # else:
        #     laplacian[i, i] += 1 * 0.1
        #     guidance_field[i] += source[i] * 0.1


    return [laplacian, guidance_field]

# linear least squares solver
def linlsq_solver(A, b, dims):
    x = linalg.spsolve(A.tocsc(),b)
    return np.reshape(x,(dims[0],dims[1]))

# stitches poisson equation solution with target
def stitch_images(source, target, dims):
    target[dims[0]:dims[1], dims[2]:dims[3],:] = source
    return target

# performs poisson blending
def blend_image(data, GRAD_MIX=False, mask_fg=None):

    equation_param = []
    ch_data = {}

    cv2.imwrite("inp_bl_mask.png", ((data['mask']) * 255).astype(np.uint8))
    # construct poisson equation
    for ch in range(3):
        ch_data['source'] = data['source'][:,:,ch]
        ch_data['mask'] = data['mask'][:,:,ch]
        ch_data['target'] = data['target'][:,:,ch]
        ch_data['dims'] = data['dims']
        equation_param.append(poisson_blending(ch_data, GRAD_MIX, mask_fg))

    # solve poisson equation
    image_solution = np.empty_like(data['source'])
    for i in range(3):
        image_solution[:,:,i] = linlsq_solver(equation_param[i][0],equation_param[i][1],data['source'].shape)

    image_solution = stitch_images(image_solution,data['target'],ch_data['dims'])

    return image_solution


def load_image(source, target, alpha, erode, mask_fg=None):
    image_data = {}

    mask = 255 - alpha.copy()

    image_data['dims'] = [0, 0]

    print(source.min(), source.max(), target.min(), target.max())
    print('In blending')
    
    cv2.imwrite("inp_source.png", ((source - target.min()) * 255 / (target.max() - target.min())).astype(np.uint8))

    mask = np.stack([mask, mask, mask], axis=2)
    source = np.stack([source, source, source], axis=2)
    target = np.stack([target, target, target], axis=2)

    # normalize the images
    image_data['source'] = source # cv2.normalize(source.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    image_data['mask']   = cv2.normalize(mask.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    image_data['target'] = target # cv2.normalize(target.astype('float'), None, 0.0, 1.0, norm_type=cv2.NORM_MINMAX)
    #   print(f"min-max {target.min()} {target.max()}")
    #   image_data['dims'] = target_offsets[int(filename[:-4])-1]

    return image_data, {'min': target.min(), 'max': target.max()}

def get_merged_depth(rendepth, monodepth, alpha, mask_fg=None, erode=True):

    image, minmax = load_image(monodepth, rendepth, alpha, erode, mask_fg)
    # if mask_bg is not None:
    #     image['target'] *= mask_bg[..., None]
    data = preprocess(image)
    final_image = blend_image(data, mask_fg=mask_fg) # blend the image
    final_depth = final_image #* (minmax['max'] - minmax['min']) + minmax['min']

    return final_depth[..., 0], image['mask']


def normalize(final_depth, min, max):
    final_depth = (final_depth - min) / (max - min)
    return final_depth
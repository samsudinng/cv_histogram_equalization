import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import skimage.io
from PIL import Image

def rgb2gray(img_array):
    rgb2gray_weights = np.array([0.2989, 0.5870, 0.1140])
    gray_img_array = np.dot(img_array, rgb2gray_weights).astype(np.uint8)

    return gray_img_array


def calc_histogram(img_array, normalize=True):
    """
    Calculate histogram from single channel image array. Since the values are integers, 
    binning method is used.
    Input:
        - img_array: 2D array of the image channel
        - normalize: if True, normalize the histogram wrt number of data points
    
    Output:
        - if normalize = False, output is histogram array
        - if normalize = True, output is pdf array
    """
    
    if img_array.dtype == np.uint8:
        num_histogram_bins = 256
    else:
        raise TypeError(f'{img_array.dtype} data type not supported!')

    if len(img_array.shape) == 3:
        img_array = rgb2gray(img_array)

    #flatten image array and calculate histogram via binning
    histogram_array = np.bincount(img_array.flatten(), minlength=num_histogram_bins)
    
    #normalize
    if normalize == True:
        num_pixels = np.sum(histogram_array)
        histogram_array = histogram_array / num_pixels
    
    return histogram_array


def calc_cdf(img_array):
    """
    Calculate cumulative distribution function from image array.
    Input:
        - img_array: 2D array of the image channel
    Output:
        - cdf: array of cdf
    """
    if len(img_array.shape) == 3:
        img_array = rgb2gray(img_array)

    pdf = calc_histogram(img_array, normalize=True)
    cdf = np.cumsum(pdf)
    return cdf, pdf #/cdf[-1]


def calculate_transform_map(img_array):
     
    if img_array.dtype == np.uint8:
        range_pixels_value = 255
    else:
        raise TypeError(f'Input data format {img_array.dtype} not supported!!')
    
    if len(img_array.shape) == 3:
        img_array = rgb2gray(img_array)

    #compute pdf, cdf
    cdf, pdf = calc_cdf(img_array)
        
    #compute old -> new intensity value transformation map
    transform_map = np.floor(range_pixels_value * (cdf / cdf[-1])).astype(img_array.dtype)

    return transform_map, cdf, pdf


def histogram_equalizer(img_array):
    
    #in case of 2 channel data (H,W), expand to 3 channel (H,W,1)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, axis=2)
    
    #shape_img = img_array.shape
    eq_img_array = np.empty_like(img_array)
    nch = eq_img_array.shape[2]

    # equalize each channel independently
    for ch in range(nch):
        # calculate equalization transform map
        ch_transform_map, ch_cdf, ch_pdf = calculate_transform_map(img_array[:,:,ch])
            
        # flatten into list
        img_list = list(img_array[:,:,ch].flatten())

        # transform values to equalize
        eq_img_list = [ch_transform_map[p] for p in img_list]

        # reshape and write back into img_array
        eq_img_array[:,:,ch] = np.reshape(np.asarray(eq_img_list), (eq_img_array.shape[0], eq_img_array.shape[1]))
            
        # save intermediate data
        if ch == 0:
            transform_map, cdf, pdf = ch_transform_map, ch_cdf, ch_pdf
        else:
            transform_map = np.vstack((transform_map, ch_transform_map))
            cdf = np.vstack((cdf, ch_cdf))
            pdf = np.vstack((pdf, ch_pdf))

    #if single channel, remove the channel dimension
    if nch == 1:
        eq_img_array = np.squeeze(eq_img_array, axis = 2)
     
    return eq_img_array, transform_map, cdf, pdf


def calculate_eme(img_array, L):

    #calculate number of blocks
    size = img_array.shape
    if len(size) == 3:
        
        rgb2gray_weights = np.array([0.2989, 0.5870, 0.1140])
        img_array = np.dot(img_array, rgb2gray_weights).astype(np.uint8)
        size = img_array.shape
        
    n_rows = np.floor(size[0]/L).astype(np.int)
    n_columns = np.floor(size[1]/L).astype(np.int)
    n_blocks = n_columns * n_rows

    #calculate EME
    cnt_block = 0
    eme = 0
    eps = np.finfo(np.float).eps
    for r in range (0, size[0]-L+1, L):
        for c in range(0,size[1]-L+1,L):
            block = img_array[r:r+L,c:c+L]#.astype(np.float)#/255

            I_max = max(np.max(block),1)
            I_min = max(np.min(block), 1)

            block_eme = 20*(np.log(I_max)-np.log(I_min))
            eme += block_eme/n_blocks
            cnt_block += 1
    
    assert cnt_block == n_blocks
    return eme




def plot_density_functions(ori_array, eq_array, legend, title, save=False, filename=None):

    ori_cdf, ori_pdf = calc_cdf(ori_array)
    eq_cdf, eq_pdf = calc_cdf(eq_array)

    plt.figure(figsize=(16,6))
    plt.suptitle(title)
    plt.subplot(121)
    plt.title('Probability density function')
    plt.plot(ori_pdf)
    plt.plot(eq_pdf)
    plt.legend(legend)
    plt.subplot(122)
    plt.title('Cumulative density function')
    plt.plot(ori_cdf)
    plt.plot(eq_cdf)
    plt.legend(legend)
    
    if save == False:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


def plot_density_functions2(ori_array, eq_array, legend, title, save=False, filename=None):

    ori_cdf, ori_pdf = calc_cdf(ori_array)
    #eq_cdf, eq_pdf = calc_cdf(eq_array)

    plt.figure(figsize=(16,6))
    plt.suptitle(title)
    plt.subplot(121)
    plt.title('Probability density function')
    plt.plot(ori_pdf)
    for arr in eq_array:
        eq_cdf, eq_pdf = calc_cdf(arr)
        plt.plot(eq_pdf)
    plt.legend(legend)
    plt.subplot(122)
    plt.title('Cumulative density function')
    plt.plot(ori_cdf)
    for arr in eq_array:
        eq_cdf, eq_pdf = calc_cdf(arr)
        plt.plot(eq_cdf)
    plt.legend(legend)
    
    if save == False:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":

    # load image
    img = Image.open('jpg/Unequalized_Hawkes_Bay_NZ.jpg')
    print(f'Original image mode: {img.mode}')
    print(f'Image size         : {img.size}')
    
    # convert to grayscale
    imgray = img.convert(mode='L')
    
    #perform histogram equalization

    # -- 3-channel RGB
    eq_rgb_array, tmap_rgb, cdf_rgb, pdf_rgb = histogram_equalizer(np.asarray(img))
    eq_rgb_skhe_array, eq_rgb_skclahe_array = skimage_equalizer(np.asarray(img))

    # --- 1 channel grayscale
    eq_gray_array, tmap, cdf, pdf = histogram_equalizer(np.asarray(imgray))
    eq_gray_skhe_array, eq_gray_skclahe_array = skimage_equalizer(np.asarray(imgray))

    # save equalized image
    eq_img = Image.fromarray(eq_rgb_array).save('eq_rgb.png')
    eq_imgray = Image.fromarray(eq_gray_array).save('eq_gray.png')
    skimage.io.imsave('eq_rgb_skimage_he.png',eq_rgb_skhe_array )
    skimage.io.imsave('eq_rgb_skimage_clahe.png',eq_rgb_skclahe_array )
    skimage.io.imsave('eq_gray_skimage_he.png',eq_gray_skhe_array )
    skimage.io.imsave('eq_gray_skimage_clahe.png',eq_gray_skclahe_array )

    #calculate EME
    eme_ori = calculate_eme(np.asarray(imgray), 8)
    eme_eq  = calculate_eme(eq_gray_array, 8)
    print(f'EME = {eme_ori} -> {eme_eq}')
    #print(f'EME = {calculate_eme(np.asarray(img),8)}')

    # plots 

    # --- density functions
    plot_density_functions(np.asarray(imgray),eq_gray_array, ['Ori','Equalized'], '',
                save=True, filename='plot_histogram_equalizer.png')
    
    plot_density_functions(eq_gray_array, eq_gray_skhe_array, ['Implementation','skimage'],
                'Comparison with skimage',
                save=True, filename='plot_vs_skimage_he.png')
    
    plot_density_functions(eq_gray_array, eq_gray_skclahe_array, ['Implementation','skimage clahe'],
                'Comparison with skimage',
                save=True, filename='plot_vs_skimage_clahe.png')
    
    ori_cdf, ori_pdf = calc_cdf(np.asarray(imgray))
    eq_cdf, eq_pdf = calc_cdf(eq_gray_array)
    
    plt.figure()
    plt.plot(ori_pdf)
    plt.plot(eq_pdf)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Distribution')
    plt.legend(['Original','Equalized'])
    plt.figure()
    plt.plot(ori_cdf)
    plt.plot(eq_cdf)
    plt.xlabel('Pixel intensity')
    plt.ylabel('Distribution')
    plt.legend(['Original','Equalized'])
    plt.show()

    

    




import numpy as np
import matplotlib.pyplot as plt
import skimage.exposure
import skimage.io

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

    pdf = calc_histogram(img_array, normalize=True)
    cdf = np.cumsum(pdf)
    return cdf, pdf #/cdf[-1]


def calculate_transform_map(img_array):
     
    if img_array.dtype == np.uint8:
        range_pixels_value = 255
    else:
        raise TypeError(f'Input data format {img_array.dtype} not supported!!')

    #compute pdf, cdf
    cdf, pdf = calc_cdf(img_array)
        
    #compute old -> new intensity value transformation map
    transform_map = np.floor(range_pixels_value * (cdf / cdf[-1])).astype(img_array.dtype)

    return transform_map, cdf, pdf


def histogram_equalizer(img_array):
    
    #calculate transform map for histogram equalization
    transform_map, cdf, pdf = calculate_transform_map(img_array)
    
    #flatten image array and convert to list, list() preserves the numpy dtype
    shape_image = img_array.shape
    all_values = list(img_array.flatten())
    
    #transform pixel values
    eq_all_values = [transform_map[p] for p in all_values]

    #reshape back into image array
    eq_img_array = np.reshape(np.asarray(eq_all_values), shape_image)
    
    return eq_img_array, transform_map, cdf, pdf


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


def generate_skimage_benchmark(img_array, method = 'RGB'):

    if method == 'RGB':

        eq_img_array = img_array.copy()
        adapt_eq_img_array = img_array.copy()

        for ch in range(img_array.shape[-1]):
            #equalize each channel independently
            eq_img_array[:,:,ch] = skimage.exposure.equalize_hist(img_array[:,:,ch])
            adapt_eq_img_array[:,:,ch] = skimage.exposure.equalize_adapthist(img_array[:,:,ch])
            
        eq_out_array = skimage.img_as_ubyte(eq_img_array)
        adapt_eq_out_array = skimage.img_as_ubyte(adapt_eq_img_array)
           
    else:
        eq_out_array = skimage.exposure.equalize_hist(img_array)
        adapt_eq_out_array = skimage.exposure.equalize_adapthist(img_array)

    return eq_out_array, adapt_eq_out_array


if __name__ == "__main__":

    #TESTING CODE
    from PIL import Image

    # load image
    img = Image.open('jpg/Unequalized_Hawkes_Bay_NZ.jpg')
    print(f'Original image mode: {img.mode}')
    print(f'Image size         : {img.size}')
    
    # convert to grayscale
    imgray = img.convert(mode='L')
    ori_img = np.asarray(imgray)
    print(f'Image array dtype  : {ori_img.dtype}')

    #perform histogram equalization
    eq_img, tmap, cdf, pdf = histogram_equalizer(ori_img)

    # create equalized image object
    eq_imgray = Image.new('L', imgray.size)
    eq_imgray = Image.fromarray(eq_img)

    
    #COMPARE WITH SOLUTION FROM SKIMAGE LIBRARY
    eq_img_skimage = skimage.exposure.equalize_hist(ori_img)
    aeq_img_skimage= skimage.exposure.equalize_adapthist(ori_img)
    
    
    # plots 

    # --- density functions
    plot_density_functions(np.asarray(imgray),np.asarray(eq_imgray), ['Ori','Equalized'], 'Comparison',
                save=False, filename='test.png')
    
    plot_density_functions(np.asarray(eq_imgray), skimage.img_as_ubyte(eq_img_skimage), ['Implementation','skimage'],
                'Comparison with skimage',
                save=False, filename='test_skimage_he.png')
    
    plot_density_functions(np.asarray(eq_imgray), skimage.img_as_ubyte(aeq_img_skimage), ['Implementation','skimage clahe'],
                'Comparison with skimage',
                save=False, filename='test_skimage_clahe.png')
    

    
    # --- save processed image
    eq_imgray.save('hist_eq.png')
    skimage.io.imsave('hist_eq_skimage.png',eq_img_skimage )
    skimage.io.imsave('adaptive_hist_eq_skimage.png',aeq_img_skimage )
    

    




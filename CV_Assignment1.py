from PIL import Image
from histogram_equalizer import histogram_equalizer, plot_density_functions2
import numpy as np
import skimage
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
import pandas as pd

img_dir = 'jpg/'
img_files = ['sample01.jpg',
             'sample02.jpeg',
             'sample03.jpeg',
             'sample04.jpeg',
             'sample05.jpeg',
             'sample06.jpg',
             'sample07.jpg',
             'sample08.jpg',
            ]

save_path = 'jpg/equalized/'
eq_modes = ['RGB', 'HSV', 'LAB'] #['RGB', 'RGB_TOTAL', 'YCbCr', 'HSV', 'LAB']
image_entropy=defaultdict(list)
#image_eme=defaultdict(list)


for f in img_files:
    
    #load image
    img = Image.open(img_dir + f)
    assert img.mode == 'RGB'

    image_entropy['FILES'].append(f)
    image_entropy['ORIGINAL'].append(img.entropy())
    #image_eme['FILES'].append(f)
    #image_eme['ORIGINAL'].append(calculate_eme(np.asarray(img),8))
    eq_imgs_array = []
    for mode in eq_modes:
        
        if mode == 'RGB':

            #RGB mode: equalize each channel independently
            eq_img_array,_,_,_ = histogram_equalizer(np.asarray(img))
            
            # equalized image
            eq_img = Image.fromarray(eq_img_array, mode='RGB')
            
            image_entropy['RGB'].append(eq_img.entropy())
            #image_eme['RGB'].append(calculate_eme(eq_img_array,8))

            eq_imgs_array.append(eq_img_array)


        elif mode == 'RGB_TOTAL':
            
            #RGB_TOTAL mode: equalize the RGB channels jointly
            #First, reshape the RGB image into one big stacked image: (H,W,3) -> (3H, W)
            img_array = np.asarray(img)
            h,w,c = img_array.shape
            img_array = np.moveaxis(img_array,-1,0).reshape(-1,w)

            #equalize the stacked image
            eq_img_array,_,_,_ = histogram_equalizer(img_array)
            
            # reshape back into RGB
            eq_img_array = np.moveaxis(eq_img_array.reshape(c,-1,w),0,-1)
           
            # equalized image
            eq_img = Image.fromarray(eq_img_array, mode='RGB')
            
            image_entropy['RGB_TOTAL'].append(eq_img.entropy())
            #image_eme['RGB_TOTAL'].append(calculate_eme(eq_img_array,8))

            eq_imgs_array.append(eq_img_array)

        elif mode == 'YCbCr':
            #convert RGB to YCbCr and extract individual channels
            ycbcr_img = img.convert('YCbCr')
            y_img,cb_img,cr_img = ycbcr_img.split()

            #equalize Y channel
            eq_y_array,_,_,_ = histogram_equalizer(np.asarray(y_img))
            
            #assemble image and convert back to RGB
            eq_img = Image.merge('YCbCr', (Image.fromarray(eq_y_array), cb_img, cr_img)).convert('RGB')
            
            image_entropy['YCbCr'].append(eq_img.entropy())
            #image_eme['YCbCr'].append(calculate_eme(np.asarray(eq_img),8))

            eq_imgs_array.append(np.asarray(eq_img))

        elif mode == 'HSV':
            #convert RGB to YCbCr and extract individual channels
            hsv_img = img.convert('HSV')
            h_img,s_img,v_img = hsv_img.split()

            #equalize V channel
            eq_v_array,_,_,_ = histogram_equalizer(np.asarray(v_img))
            
            #assemble image and convert back to RGB
            eq_img = Image.merge('HSV', (h_img, s_img, Image.fromarray(eq_v_array))).convert('RGB')
            
            image_entropy['HSV'].append(eq_img.entropy())
            #image_eme['HSV'].append(calculate_eme(np.asarray(eq_img),8))

            eq_imgs_array.append(np.asarray(eq_img))
        
        elif mode == 'LAB':
            #convert RGB to LAB and re-scale L channel to [0, 255] range
            lab_img = skimage.color.rgb2lab(np.asarray(img))
            L_array = (np.clip(lab_img[:,:,0], 0.0, 100.0)*255/100).astype(np.uint8)
            
            #equalize L channel
            eq_L_array,_,_,_ = histogram_equalizer(np.asarray(L_array))

            #convert back to [0,100] range
            lab_img[:,:,0] = eq_L_array.astype(np.float64)*100/255
            eq_img_array = (skimage.color.lab2rgb(lab_img)*255).astype(np.uint8)
            eq_img = Image.fromarray(eq_img_array, mode='RGB')
            
            image_entropy['LAB'].append(eq_img.entropy())
            #image_eme['LAB'].append(calculate_eme(np.asarray(eq_img),8))

            eq_imgs_array.append(np.asarray(eq_img))
            
        #save equalized image
        filename = f.split('.')[0]
        eq_img.save(f'{save_path}{filename}_eq_{mode}.png')
        
    #plot histogram density
    all_legend = ['Original']
    for mode in eq_modes:
        all_legend.append(mode+'_Eq')
    plot_density_functions2(np.asarray(img),eq_imgs_array, all_legend, f,
                save=True, filename=f'plot/{filename}_eq_cdf.png')
        

image_entropy_df = pd.DataFrame(image_entropy)
#image_eme_df = pd.DataFrame(image_eme)
print(image_entropy_df)
#print(image_eme_df)
image_entropy_df.to_csv(save_path+'equalized_entropy.csv')
#image_eme_df.to_csv(save_path+'equalized_eme.csv')

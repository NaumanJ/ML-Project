------------------------------------------------------------------------------------------------------------
Measuring Texture Feature Coarseness
1. Input Image
Start with the input image.
2. Define Neighborhood Sizes
Define a set of neighborhood sizes 𝑘k (e.g., 1𝑥1,2𝑥2,4𝑥4,…1x1,2x2,4x4,…).
3. Compute Local Averages
For each pixel (𝑖,𝑗)(i,j) in the image, compute the average gray level 𝐴𝑘(𝑖,𝑗)Ak​(i,j) over neighborhoods of size 2𝑘×2𝑘2k×2k.
4. Compute Differences
For each neighborhood size 𝑘k:
Compute the horizontal difference 𝐷ℎ(𝑖,𝑗,𝑘)=∣𝐴𝑘(𝑖+2𝑘,𝑗)−𝐴𝑘(𝑖−2𝑘,𝑗)∣Dh​(i,j,k)=∣Ak​(i+2k,j)−Ak​(i−2k,j)∣.
Compute the vertical difference 𝐷𝑣(𝑖,𝑗,𝑘)=∣𝐴𝑘(𝑖,𝑗+2𝑘)−𝐴𝑘(𝑖,𝑗−2𝑘)∣Dv​(i,j,k)=∣Ak​(i,j+2k)−Ak​(i,j−2k)∣.
5. Determine Maximum Difference
For each pixel (𝑖,𝑗)(i,j):
Find the neighborhood size 𝑘k that maximizes the sum of the differences 𝐷ℎDh​ and 𝐷𝑣Dv​:𝑆(𝑖,𝑗)=2𝑘𝑚𝑎𝑥S(i,j)=2kmax​
Where 𝑘𝑚𝑎𝑥kmax​ is the scale that gives the maximum 𝐷ℎ+𝐷𝑣Dh​+Dv​.
6. Compute Average Coarseness
Average the values of 𝑆(𝑖,𝑗)S(i,j) over all pixels (𝑖,𝑗)(i,j) in the image to obtain the coarseness feature 𝐹𝑐𝑜𝑎𝑟𝑠𝑒Fcoarse​:𝐹𝑐𝑜𝑎𝑟𝑠𝑒=1𝑁∑𝑖,𝑗𝑆(𝑖,𝑗)Fcoarse​=N1​i,j∑​S(i,j)
Where 𝑁N is the total number of pixels.
------------------------------------------------------------------------------------------------------------  
import numpy as np
from skimage import io, color

def compute_coarseness(image):
    """
    Compute the coarseness of an image based on Tamura's texture features.
    
    Parameters:
    image (numpy array): Input image in grayscale or RGB format.
    
    Returns:
    float: Coarseness value.
    """
    # Convert image to grayscale if it is in RGB format
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    
    # Normalize the image to the range [0, 1]
    image = image / image.max()
    
    # Define different window sizes (scales) to analyze the texture
    scales = [1, 2, 4, 8, 16]
    
    # Initialize arrays to store horizontal and vertical differences
    E_h = np.zeros((len(scales), image.shape[0], image.shape[1]))
    E_v = np.zeros((len(scales), image.shape[0], image.shape[1]))
    
    for i, scale in enumerate(scales):
        # Calculate horizontal and vertical differences for each scale
        if image.shape[0] > scale and image.shape[1] > scale:
            E_h[i, scale:-scale, scale:-scale] = np.abs(
                image[scale:-scale, 2*scale:] - image[scale:-scale, :-2*scale]
            ) / (2 * scale)
            E_v[i, scale:-scale, scale:-scale] = np.abs(
                image[2*scale:, scale:-scale] - image[:-2*scale, scale:-scale]
            ) / (2 * scale)
    
    # Compute the maximum difference for each pixel across all scales
    E = np.maximum(E_h, E_v)
    
  # Select the scale with the maximum difference for each pixel
    S_best = np.argmax(E, axis=0)
    
    # Compute the average best scale as the coarseness measure
    coarseness = np.mean(scales[S_best])
    
    return coarseness
------------------------------------------------------------------------------------------------------------
Function Code Explanation:
NDVI Calculation: The calculate_ndvi function computes the NDVI using the NIR and Red bands.
Band Loading: The load_bands function loads the NIR and Red bands from the TIFF image.
NDVI Saving: The save_ndvi function saves the NDVI image as a new TIFF file, preserving the original image's metadata.
NDVI Plotting: The plot_ndvi function visualizes the NDVI image using matplotlib.
Main Function: The main function orchestrates loading the bands, calculating NDVI, saving the NDVI image, and plotting it.
------------------------------------------------------------------------------------------------------------
def calculate_ndvi(nir, red):
    """Calculate NDVI from NIR and Red bands."""
    ndvi = (nir - red) / (nir + red)
    return ndvi

def load_bands(image_path, nir_band_index, red_band_index):
    """Load NIR and Red bands from a multi-band UAV TIFF image."""
    with rasterio.open(image_path) as src:
        nir = src.read(nir_band_index)
        red = src.read(red_band_index)
    return nir, red

def save_ndvi(ndvi, profile, output_path):
    """Save NDVI image as a new TIFF file."""
    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress='lzw'
    )
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(ndvi.astype(rasterio.float32), 1)

def plot_ndvi(ndvi):
    """Plot NDVI image."""
    plt.figure(figsize=(10, 10))
    plt.imshow(ndvi, cmap='RdYlGn')
    plt.colorbar(label='NDVI')
    plt.title('NDVI Image')
    plt.axis('off')
    plt.show()

def main(image_path, nir_band_index, red_band_index, output_path):
    # Load NIR and Red bands
    nir, red = load_bands(image_path, nir_band_index, red_band_index)
    
    # Calculate NDVI
    ndvi = calculate_ndvi(nir, red)
    
    # Load image metadata to use for saving the NDVI image
    with rasterio.open(image_path) as src:
        profile = src.profile  


    # Save NDVI image
    save_ndvi(ndvi, profile, output_path)
    
    # Plot NDVI image
    plot_ndvi(ndvi)

if __name__ == "__main__":
    image_path = 'path_to_your_uav_image.tiff'  # Replace with the path to your UAV TIFF image
    nir_band_index = 4  # Replace with the index of the NIR band (1-based index)
    red_band_index = 3  # Replace with the index of the Red band (1-based index)
    output_path = 'path_to_output_ndvi_image.tiff'  # Path to save the NDVI image
    
    main(image_path, nir_band_index, red_band_index, output_path)
 ------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------


Useful blocks to build Unet

conv - BN - Activation - conv - BN - Activation - Dropout (if enabled)

'''
def conv_block(x, filter_size, size, dropout, batch_norm=False):
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)
    
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv

-------------------------------------------------------------------------------------------------------


def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    '''
    Residual convolutional layer.
    Two variants....
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).
    1. conv - BN - Activation - conv - BN - Activation  - shortcut  - BN - shortcut+BN                                         
    2. conv - BN - Activation - conv - BN - shortcut  - BN - shortcut+BN - Activation                                     
    '''
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)
    
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)

 if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)    #Activation after addition with shortcut (Original residual block)
    return res_path

------------------------------------------------------------------------------------------------------

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x)
    shape_g = K.int_shape(gating)
# Getting the x signal to the same shape as the gating signal
    theta_x = layers.Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)
# Getting the gating signal to the same number of filters as the inter_shape
    phi_g = layers.Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                        strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                                 padding='same')(phi_g)  # 16
    concat_xg = layers.add([upsample_g, theta_x])
    act_xg = layers.Activation('relu')(concat_xg)
    psi = layers.Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = layers.Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = repeat_elem(upsample_psi, shape_x[3])
    y = layers.multiply([upsample_psi, x])
    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = layers.BatchNormalization()(result)
    return result_bn
def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x
------------------------------------------------------------------------------------------------------

def repeat_elem(tensor, rep):
    # lambda function to repeat Repeats the elements of a tensor along an axis
    #by a factor of rep.
    # If tensor has shape (None, 256,256,3), lambda will return a tensor of shape 
    #(None, 256,256,6), if specified axis=3 and rep=2.

     return layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

------------------------------------------------------------------------------------------------------

def UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    

    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
   
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
   
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
   
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model 
    model = models.Model(inputs, conv_final, name="UNet")
    print(model.summary())
    return model
------------------------------------------------------------------------------------------------------

def Attention_UNet(input_shape, NUM_CLASSES=1, dropout_rate=0.0, batch_norm=True):
    '''
    Attention UNet, 
    
    '''
    # network structure
    FILTER_NUM = 64 # number of basic filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, convolution + pooling
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)
    # DownRes 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)
    # DownRes 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)
    # DownRes 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)
    # DownRes 5, convolution only
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

# Upsampling layers
    # UpRes 6, attention gated concatenation + upsampling + double residual convolution
    gating_16 = gating_signal(conv_8, 8*FILTER_NUM, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*FILTER_NUM)
    up_16 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    gating_32 = gating_signal(up_conv_16, 4*FILTER_NUM, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*FILTER_NUM)
    up_32 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    gating_64 = gating_signal(up_conv_32, 2*FILTER_NUM, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*FILTER_NUM)
    up_64 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
    gating_128 = gating_signal(up_conv_64, FILTER_NUM, batch_norm)
    att_128 = attention_block(conv_128, gating_128, FILTER_NUM)
    up_128 = layers.UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    conv_final = layers.Conv2D(NUM_CLASSES, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    return model

------------------------------------------------------------------------------------------------------

    
















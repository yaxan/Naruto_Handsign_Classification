import tensorflow as tf

def otsu_thresholding(image):
    image = tf.convert_to_tensor(image, name="image")

    rank = image.shape.rank
    if rank != 2 and rank != 3:
        raise ValueError("Image should be either 2 or 3-dimensional.")

    if image.dtype!=tf.int32:
        image = tf.cast(image, tf.int32)

    r, c = image.shape
    hist = tf.math.bincount(image, dtype=tf.int32)
    
    if len(hist)<256:
        hist = tf.concat([hist, [0]*(256-len(hist))], 0)

    current_max, threshold = 0, 0
    total = r * c

    spre = [0]*256
    sw = [0]*256
    spre[0] = int(hist[0])

    for i in range(1,256):
        spre[i] = spre[i-1] + int(hist[i])
        sw[i] = sw[i-1]  + (i * int(hist[i]))

    for i in range(256):
        if total - spre[i] == 0:
            break

        meanB = 0 if int(spre[i])==0 else sw[i]/spre[i]
        meanF = (sw[255] - sw[i])/(total - spre[i])
        varBetween = (total - spre[i]) * spre[i] * ((meanB-meanF)**2)

        if varBetween > current_max:
            current_max = varBetween
            threshold = i

    final = tf.where(image>threshold,255,0)
    return final
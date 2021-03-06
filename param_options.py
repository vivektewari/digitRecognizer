
class Model1:
    # layer 28,28, pa 29,29 c 27,27 p 9,9 ,pa 10,10 c 5,5 p 1,1

    channels = [30, 10]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [3, 6]
    pads = [1, 1]
    strides = [1, 1]
    pools = [3, 5]
    # layer 28,28, pa 28,28 c 21,21 p 7,7 ,pa 7,7 c 5,5 p 1,1
    channels = [10, 10]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [8, 3]
    pads = [1, 0]
    strides = [1, 1]
    pools = [3, 5]
# best
    channels = [31,65]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5]#5,5,3
    pads = [0, 0,1]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [256, 10]#1040
class Model2:
    channels = [31,65]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5]#5,5,3
    pads = [0, 0,0]
    strides = [1,1, 1,1]
    pools = [2, 2,1]
    fc1_p = [None, 10]#1040
class Model3:
    channels = [31,65,128]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [3 ,3,2]#5,5,3
    pads = [0, 0,0,0]
    strides = [1,1, 1,1]
    pools = [2, 2,2,1]
    fc1_p = [256, 14]
class Model3_p:
    channels = [31,65,128]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,5,2]#5,5,3
    pads = [0, 0,0,0]
    strides = [1,1, 1,1]
    pools = [2, 2,2,1]
    fc1_p = [None,700*(14+1)]
class Model3_p_28:
    channels = [31,65,128]
    input_image_dim = (28, 28)
    start_channel = 1
    convs = [5 ,3,2]#5,5,3
    pads = [0, 0,0,0]
    strides = [1,1, 1,1]
    pools = [2, 2,2,1]
    fc1_p = [None,45*(14+1)]
class Model3_p_112:
    channels = [31,65,128]
    input_image_dim = (28*4, 28*4)
    start_channel = 1
    convs = [5 ,5,3,3,2]#5,5,3
    pads = [0, 0,0,0,0]
    strides = [1,1, 1,1,1]
    pools = [2, 2,2,2,1]#receptive 6,
    fc1_p = [None,1100*(14+1)]#adding more grid
class Model3_p_112_large:
    channels = [31,65,128,128,128,256,256,512]
    input_image_dim = (28*4, 28*4)
    start_channel = 1
    convs = [5 ,5,3,2,2,2,2,2]#5,5,3
    pads = [0, 0,0,0,0,0,0,0,0,0,0]
    strides = [1,1, 1,1,1,1,1,1,1]
    pools = [2, 2,2,2,1,1,1,1,1,1]#receptive 6,
    fc1_p = [None,1100*(14+1)]#adding more grid
class Model4:
        channels = [11, 22]
        input_image_dim = (28, 28)
        start_channel = 1
        convs = [5, 3, 2,2]  # 5,5,3
        pads = [0, 0, 0, 0,0]
        strides = [1, 1, 1, 1,1]
        pools = [1, 1, 1, 1,1]  # [2, 2,2,1]
        fc1_p = [64, 14]  # [256, 14]


class Model3_112:
    channels = [11, 22]
    input_image_dim = (28*4, 28*4)
    start_channel = 1
    convs = [5, 3, 2, 2]  # 5,5,3
    pads = [0, 0, 0, 0, 0]
    strides = [1, 1, 1, 1, 1]
    pools = [1, 1, 1, 1, 1]  # [2, 2,2,1]
    fc1_p = [64, 14]  # [256, 14]
class Model3_112_2:
    channels = [16,32, 64,128,256,512,1024]
    input_image_dim = (28*4, 28*4)
    start_channel = 1
    convs = [5, 5, 3, 3,2,2,2]  # 5,5,3
    pads = [0, 0, 0, 0, 0,0,0,0]
    strides = [1, 1, 1, 1, 1,1,1]
    pools = [2, 2, 2, 2, 1,1,1]  # [2, 2,2,1]
    fc1_p = [128, 14]  # [256, 14]
class ModelSSD:
    channels = [16,32, 64, 128, 256, 512, 512]
    input_image_dim = (28 * 4, 28 * 4)
    start_channel = 1
    convs = [5, 5,2, 3, 2, 2, 2]  # 5,5,3
    pads = [0, 0, 0, 0, 0, 0, 0, 0]
    strides = [1, 1, 1, 1, 1, 1, 1]
    pools = [2, 2, 2, 2, 1, 1, 1]  # [2, 2,2,1]
    fc1_p = [None, 700*(14+1)]  # [256, 14]



class DataLoad1:
    # data_frame = None
    label = 'label'
    reshape_pixel = (28, 28)
    pixel_col = ['pixel' + str(i) for i in range(reshape_pixel[0] * reshape_pixel[1])]
    path = ""

class DataLoad1_l(DataLoad1):
    label = 'label'
    reshape_pixel = (28, 28)
    pixel_col = ['pixel' + str(i) for i in range(reshape_pixel[0] * reshape_pixel[1])]
    path = ""
    localization_col = 'localisation'
class DataLoad1_l_112(DataLoad1):
    label = 'label'
    reshape_pixel = (28*4, 28*4)
    pixel_col = ['pixel' + str(i) for i in range(reshape_pixel[0] * reshape_pixel[1])]
    path = ""
    localization_col = 'localisation'
class DataLoad1_l2_112(DataLoad1):
    label = 'label'
    reshape_pixel = (28*4, 28*4)
    pixel_col = ['pixel' + str(i) for i in range(reshape_pixel[0] * reshape_pixel[1])]
    path_loc = ""
    path = ""
    localization_col = 'localisation'







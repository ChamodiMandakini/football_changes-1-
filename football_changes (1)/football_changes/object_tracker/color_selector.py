'''
Let the user select a specific color and gather hsv values according
to that specific color and  create masks according to the selected
colors and percentage
'''
try:
    import sys
    import cv2 as cv
    import numpy as np
except Exception as e:
    print(f"Error\n\t{e}")
    sys.exit()


def getUniqueColor(img):
    '''Return numpy array of unique colors and its color.
    arguments:
        img -> any image
    output:
        color -> a numpy array with unique colors BGR/RGB format
                np.array([[  0   0   0]
                          [  0   0   1]
                          [255 255 255]])
        color_count -> a count of each of these unique colors
    '''
    try:
        a = np.copy(img[:, :, :])
        x = np.split(a.ravel(), img.shape[0]*img.shape[1])
    except Exception as e:
        print(f"Error\n\t{e}")
        raise NotImplementedError
        sys.exit()
    color, color_count = np.unique(x, return_counts=True, axis=0)
    return color, color_count


def percentageColor(img, black=True):
    '''Return the percentage of color with respect to black color
    arguments:
        img -> any image
        black -> if black is TRUE then RETURNS percentage of black in
                 the image else RETURNS percentage of other color
    NOTE: THIS METHOD CAN ONLY BE APPLIED FOR MASKED IMAGES, MASKED IMAGES
          CONTAINS ONLY TWO COLORS(BLACK & ANOTHER COLOR)
    '''
    if type(black) != bool:
        raise ValueError("black argument should be True or False")
        sys.exit()
    try:
        color, color_count = getUniqueColor(img)
    except Exception as e:
        print(f"Error\n\t{e}")
        raise NotImplementedError
        sys.exit()
    # black is the lowest color
    black_count = color_count[0]
    # percentage of black color
    per_of_black = black_count/(img.shape[0]*img.shape[1])
    if black:
        return per_of_black * 100
    else:
        return (1 - per_of_black)*100


def getRange(hsv):
    '''Returns the lowerbound and the upperbound
    HSV values
    arguments:
        hsv -> hsv of the color
    output:
        lowerbound & upperbound
        Both of these variables will be (3x1)numpy array
            np.array([6,22,255])
    '''
    upper_saturation = 255
    lower_saturation = 50
    upper_value = 255
    lower_value = 20
    upper_bound = np.array([0, upper_saturation, upper_value])
    lower_bound = np.array([0, lower_saturation, lower_value])
    current_hue = int(hsv[0] / 2)
    if 6 >= current_hue >= 0:
        upper_bound[0] = 6
        lower_bound[0] = 0
    elif 22 >= current_hue > 6:
        upper_bound[0] = 22
        lower_bound[0] = 7
    elif 36 >= current_hue > 22:
        upper_bound[0] = 36
        lower_bound[0] = 23
    elif 75 >= current_hue > 36:
        upper_bound[0] = 75
        lower_bound[0] = 37
    elif 109 >= current_hue > 76:
        upper_bound[0] = 109
        lower_bound[0] = 77
    elif 136 >= current_hue > 109:
        upper_bound[0] = 136
        lower_bound[0] = 110
    elif 147 >= current_hue > 136:
        upper_bound[0] = 147
        lower_bound[0] = 137
    elif 175 >= current_hue > 147:
        upper_bound[0] = 175
        lower_bound[0] = 148
    elif 180 >= current_hue > 175:
        upper_bound[0] = 180
        lower_bound[0] = 176
    else:
        raise NotImplementedError
    return upper_bound, lower_bound


def colorCheck(img, color_name):
    '''Apply masks for each color and check its percentage
    argument:
        img -> any image
        color_name -> color that you need to apply masks for ..
    '''
    try:
        high, low = getRange(color_name)
    except KeyError:
        print("Color not Implemented")
        raise NotImplementedError
    hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    mask = cv.inRange(hsv, low, high)
    result = cv.bitwise_and(img, img, mask=mask)
    return percentageColor(result, False)


def colorSelector(img, teamcolors):
    '''Selects the best color for the image (img)
    argument:
        img -> any image
        teamcolors -> colors of teams that the teams are playing
                    order should be [team1,team2,refree]
                    eg: ([[[0,10,5]]],[[[10,42,12]]],[[[0,45,0]]])
                    type of each element: np.uint8
    variables:
        cls -> name of the class (team1 or team2 or refree)
        bestcolor -> the best possible color
        position -> position of the bestcolor
    '''
    _cls = ""
    bestcolor = [colorCheck(img, i) for i in teamcolors]
    position = bestcolor.index(max(bestcolor))
    if position == 2:
        _cls = 'refree'
    else:
        _cls = f"team{position+1}"
    return bestcolor[position], _cls

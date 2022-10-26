import hashlib
import hmac
import base64
import requests
import urllib.parse as urlparse
import os
import math
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2
import subprocess
import tensorflow_addons as tfa
import argparse
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input as preprocess_input_xception
from glob import glob

api_key = 'AIzaSyB-XTkheRppWGlcJ8tbEUwDNRksZ37Iyi4'
secret = 'QFkeZ_WWEa5U2-LkoxbvMoWG1_E='


def sign_url(input_url=None, secret=None):
    """
    Sign a request URL with a URL signing secret.

    Parameter input_url: The URL to sign
    Precondition: input_url is a string

    Parameter secret: Your URL signing secret
    Precondition: secret is a string

    Returns:
    The signed request URL, string
  """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()


def convert_params(parameter_dict):
    """
    Given a dictionary with keys of Maps Static API parameter and values
    corresponding to the parameter type, returns the string added to the end
    of the URL to make a query

    Parameter parameter_dict: a dictionary with keys and values described above
    Precondition: parameter_dict is a non-empty dictionary

    Returns: URL addon with parameters [string]
    """
    return_str = ''
    for key, value in parameter_dict.items():
        return_str += f'{key}={value}&'
    return return_str[:-1]


def develop_url(lat, lng, zoom=18):
    """
    Given a latitude, longitude, and zoom level (as specified in Maps Static API),
    generates a URL that redirects to the image with the given arguments

    Parameter lat: latitude
    Precondition: lat is a float or int

    Parameter lng: longitude
    Precondition: lng is a float or int

    Parameter zoom: zoom level for Maps Static API
    Precondition: zoom is an in from 1 to 21

    Returns: signed URL that links to image, string
    """
    url = 'https://maps.googleapis.com/maps/api/staticmap?'
    params = {
        'center': f'{lat},{lng}',
        'size': '512x512',
        'maptype': 'satellite',
        'zoom': str(zoom),  # '18'
        'key': api_key
    }
    url = f'{url}{convert_params(params)}'
    return sign_url(url, secret)


def save_img(lat, lng, zoom, fn):
    """
    Given a latitude, longitude, zoom level, and filename to save the image as,
    generates a signed URL to get the corresponding image and applies preprocessing.
    The preprocessing is removing the bottom 15 pixels and resizing the image to
    512 by 512. The image is saved to disk according to the file name

    Parameter lat: latitude
    Precondition: lat is a float or int

    Parameter lng: longitude
    Precondition: lng is a float or int

    Parameter zoom: zoom level for Maps Static API
    Precondition: zoom is an in from 1 to 21

    Parameter fn: file name to give the image
    Precondition: fn is a string

    Returns: None
    """
    url = develop_url(lat, lng, zoom)
    r = requests.get(url)
    f = open(fn, 'wb')
    f.write(r.content)
    f.close()
    img = image.img_to_array(image.load_img(fn))
    img = img[:-15, :, :]
    img = cv2.resize(img, (512, 512))
    image.save_img(fn, img)


def get_pixel_coord(lat, lng, zoom):
    """
    Given a latitude and longitude, generates a corresponding x and y for a flat
    world view based on zoom level. The dimensions of this world view is
    (512*2^zoom level, 512 because it is the size of images used.)

    Parameter lat: latitude
    Precondition: lat is a float or int

    Parameter lng: longitude
    Precondition: lng is a float or int

    Parameter zoom: zoom level for Maps Static API
    Precondition: zoom is an in from 1 to 21

    Returns: (x, y), a Tuple[float] that corresponds to flat world view
    """
    # applies mercator projection to lat and lon to get flat "world view"
    size = 512
    x = (lng+180)*(size/360)
    # convert from degrees to radians
    latRad = lat*math.pi/180
    # get y value
    mercN = math.log(math.tan((math.pi/4)+(latRad/2)))
    y = (size/2)-(size*mercN/(2*math.pi))

    scale = 1 << zoom  # (2**zoom)
    return x*scale, y*scale


def rev_to_lat_lng(pix_x, pix_y, zoom):
    """
    From pixel coords of zoomed world view get lng and lat
    (applying reverse scaling and mercator proj)

    Parameter pix_x: x value in flat world view
    Precondition: pix_x is a float

    Parameter pix_y: y value in flat world view
    Precondition: pix_y is a float

    Parameter zoom: zoom level for Maps Static API
    Precondition: zoom is an in from 1 to 21

    Returns: (lat, lng), a Tuple[float] that corresponds to latitude and longitude
    """
    size = 512
    worldx, worldy = math.exp(math.log(pix_x)-zoom*math.log(2)), \
        math.exp(math.log(pix_y)-zoom*math.log(2))
    lng = worldx / (size / 360) - 180
    temp = ((worldy - (size / 2))*-2*math.pi)/size
    temp = math.exp(temp)
    temp = math.atan(temp)  # atan2?
    lat = (temp - math.pi/4)*2*180/math.pi
    return lat, lng


def img_scrape_surr(lat, lng, n, m, zoom=18):
    """
    Produces n by m images (n rows, m cols) centered around given latitude and longitude
    with given zoom level. Also merges all these n by m images into a merged image

    Parameter lat: latitude
    Precondition: lat is a float or int

    Parameter lng: longitude
    Precondition: lng is a float or int

    Parameter n: number of rows of images
    Precondition: n is an int > 0

    Parameter m: number of cols of images
    Precondition: m is an int > 0

    Parameter zoom: zoom level for Maps Static API
    Precondition: zoom is an in from 1 to 21

    Returns: None
    """
    x = np.indices((n, m)).astype(float)
    x[0] -= n//2
    x[1] -= m//2
    if n % 2 == 0:
        x[0] += 0.5
    if m % 2 == 0:
        x[1] += 0.5
    x *= 512*2  # size*2

    dirname = os.path.join('outputs', f'{lat:.6f}_{lng:.6f}_{n:d}by{m:d}')
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    pix_x, pix_y = get_pixel_coord(lat, lng, zoom)
    for i in range(n):  # row
        for j in range(m):  # col
            n_pix_x = pix_x + x[1][i][j]
            n_pix_y = pix_y + x[0][i][j]
            n_lat, n_lng = rev_to_lat_lng(n_pix_x, n_pix_y, zoom)
            fn = os.path.join(
                'outputs', f'{lat:.6f}_{lng:.6f}_{n:d}by{m:d}/{i:d}_{j:d}.png')
            save_img(n_lat, n_lng, zoom, fn)

    img_files = glob(os.path.join(dirname, '*.png'))
    get_merged_img(dirname, img_files, 'merged.png', n, m)


def run_yolo(dirname, n, m, model_fn, conf):
    """
    Runs YOLO model on grid of images and generates data csv file that contains 
    information about area covered by each class

    Parameter dirname: file path to folder containing images
    Precondition: dirname is a string

    Parameter n: number of rows of images
    Precondition: n is an int > 0

    Parameter m: number of cols of images
    Precondition: m is an int > 0

    Parameter model_fn: name of model file name 
    Precondition: model_fn is a string

    Parameter conf: confidence value for yolo model
    Precondition: is a number between 0 and 1 inclusive, or None

    Returns: None (generates csv output)
    """
    if conf is None:
        conf = 0.1
    bashCom = "python yolov5/detect.py --weights "+model_fn+" --img 512 --conf " + \
        str(conf) + " --project " + dirname + \
        " --name yolo --save-txt --source " + dirname
    process = subprocess.Popen(bashCom.split(), stdout=subprocess.PIPE)
    process.communicate()
    # get most recent yolo run
    dirname2 = glob(os.path.join(dirname, '*/'))[-1]
    get_merged_img(dirname2, glob(os.path.join(dirname2, '*.png')),
                   'merged_indiv.png', n, m)

    dirname3 = os.path.join(dirname2, 'labels')
    txt_files = glob(os.path.join(dirname3, '*.txt'))[:-1]
    get_area(txt_files, n, m, dirname2)
    return


def get_area(txt_files, n, m, dirname):
    """
    Given the bounding boxes generated by the YOLO model output in txt format, 
    calculates the area covered by each class for each cell image and 
    writes the output to a csv.

    Parameter txt_files: a list of txt file names that contain YOLO bounding box info
    Precondition: txt_files is List[string...]

    Parameter n: number of rows of images
    Precondition: n is an int > 0

    Parameter m: number of cols of images
    Precondition: m is an int > 0

    Parameter dirname: file path to YOLO output folder w/ annotated images
    Precondition: dirname is a string

    Returns: None (generates a csv file with areas)
    """
    # key = file name, value = list of [label, x, y, w, h] where x and y are centered
    d = {}
    for t in txt_files:
        with open(t) as f:
            lines = f.readlines()
            lines = [l.strip().split() for l in lines]
        d[t] = lines
    for t in txt_files:
        d[t] = get_area_helper(d[t])
    A = np.zeros((n*m, 7))  # number of individual images x classes
    for t in txt_files:
        n1, m1 = [int(x) for x in os.path.split(t)
                  [-1].split('.')[0].split('_')]
        for c in d[t]:
            A[n1*m+m1, int(c)] = d[t][c]

    avg_sums = A.sum(axis=0)/(n*m)
    A = np.vstack((A, avg_sums))
    # 0 paved parking, 3 fac w truck, 5 unpaved
    idx_to_name = {0: 'paved parking lot', 2: 'large parking lot',
                   1: 'unpaved parking lot', 4: 'unpaved area with trucks', 5: 'unpaved area',
                   3: 'facility with trucks', 6: 'airport facilities'}
    index_names = []
    for y in range(n):
        for x in range(m):
            index_names.append(str(y)+'_'+str(x))
    index_names.append('avg sum')
    df = pd.DataFrame(A, columns=dict(
        sorted(idx_to_name.items())).values(), index=index_names)
    df.to_csv(os.path.join(dirname, 'area.csv'))


def get_area_helper(l2):
    """
    Given a 2D list where each element corresponds to a different bounding box in
    the format [class, center x, center y, width, height], calculate the percent
    area covered by each class. Avoids double counting within the same class and 
    prioritizes smaller bouding boxes (less area) across different classes.

    Parameter l2: 2D list of bounding boxes in one image
    Precondition: l2 is a List[List[int, float, float, float, float]]

    Returns: Dict[int: float], where keys are classes and values are area covered 
    by that class
    """
    x = []
    for l in l2:
        for i in range(len(l)):
            l[i] = float(l[i])
        coords, area = get_x_y(l[1:])
        x.append([l[0]]+coords+[area])

    # in theory can change this to prioritize any metric you want
    # if you want to give priority to confidence interval, you first have to
    # change YOLO model output to give confidence values, then sort on this
    x.sort(key=lambda x: x[-1], reverse=True)
    Ax, Ay = 1024, 1024
    A = np.zeros((Ax, Ay))
    A -= 1
    for bbox in x:
        A[int(bbox[3]*Ay):int(bbox[4]*Ay),
          int(bbox[1]*Ax):int(bbox[2]*Ax)] = bbox[0]
    d = {}
    unique, counts = np.unique(A, return_counts=True)
    counts = dict(zip(unique, counts))
    for i in range(7):  # 7 labels
        if i in counts:
            d[i] = counts[i]/(Ax*Ay)
    return d


def get_x_y(l):
    """
    Given a list of center x, center y, width, height (0 to 1) corresponding to
    a bounding box, generates a new list of [left x, right x, top y, bottom y] 
    and the area covered by this bounding box

    Parameter l: bounding box information list
    Precondition: List[float, float, float, float]

    Returns: new List[float, float, float, float] of the boundary of bounding box,
    area (a float)
    """
    x, y, w, h = l
    return [x-w/2, x+w/2, y-h/2, y+h/2], w*h


def run_multiclass(n1, n2, n, m, dirname, model_fn, out_name):
    """
    Generates outputs and statistics about the results from running multilabel 
    model on the n by m images.

    Parameter n1: a numpy array with binary values
    Precondition: np.array of size (n*m, 7) with either 0 or 1

    Parameter n2: a numpy array with float values
    Precondition: np.array of size (n*m, 7) with floats between 0 and 1 inclusive

    Parameter n: number of rows of images
    Precondition: n is an int > 0

    Parameter m: number of cols of images
    Precondition: m is an int > 0

    Parameter dirname: file path to folder containing images
    Precondition: dirname is a string

    Parameter model_fn: file name of model
    Precondition: model_fn is a string 

    Parameter out_name: desired folder name where outputs go
    Precondition: out_name is a string

    Returns: None (generates multiple csv/xslx files)
    """
    newdirname = make_folder(dirname, model_fn, out_name)
    d1 = dict()
    d2 = dict()
    idx_to_name = {0: 'paved parking lot', 1: 'large parking lot',
                   2: 'unpaved parking lot', 3: 'unpaved area with trucks', 4: 'unpaved area',
                   5: 'facility with trucks', 6: 'airport facilities'}
    # print(n1, n2)
    for x in range(n1.shape[1]):
        d1[idx_to_name[x]] = n1[:, x].reshape((n, m))
        d2[idx_to_name[x]] = n2[:, x].reshape((n, m))

    # upper bound count (number of cells)
    n1sum = n1.sum(axis=0)
    n1percent = n1sum/(n*m)

    # lower bound count (number of connected components)
    connected_counts = []
    for x in range(n1.shape[1]):
        arr = np.copy(d1[idx_to_name[x]])
        count = num_conn_comps(arr)
        connected_counts.append(count)
    n1_lsum = np.asarray(connected_counts)
    n1_lpercent = n1_lsum/(n*m)

    n3 = np.vstack([n1sum, n1percent, n1_lsum, n1_lpercent])
    index_names = ['number total cells', 'percent total cells',
                   'number connected components', 'percent connected components']
    df = pd.DataFrame(n3, columns=dict(
        sorted(idx_to_name.items())).values(), index=index_names)
    df.to_csv(os.path.join(newdirname, 'stats.csv'))

    path1 = os.path.join(newdirname, 'predictions_rounded.csv')
    path2 = os.path.join(newdirname, 'predictions.csv')
    path3 = os.path.join(newdirname, 'pred_rounded_categories.xlsx')
    path4 = os.path.join(newdirname, 'pred_categories.xlsx')
    np_to_csv(n1, path1, idx_to_name)
    np_to_csv(n2, path2, idx_to_name)
    np_class_dict_to_csv(d1, path3)
    np_class_dict_to_csv(d2, path4)
    return


def run_binary(n1, n2, n, m, dirname, model_fn, out_name):
    """
    Generates outputs and statistics about the results from running multilabel 
    model on the n by m images

    Parameter n1: a numpy array with binary values
    Precondition: np.array of size (n*m, 7) with either 0 or 1

    Parameter n2: a numpy array with float values
    Precondition: np.array of size (n*m, 7) with floats between 0 and 1 inclusive

    Parameter n: number of rows of images
    Precondition: n is an int > 0

    Parameter m: number of cols of images
    Precondition: m is an int > 0

    Parameter dirname: file path to folder containing images
    Precondition: dirname is a string

    Parameter model_fn: file name of model
    Precondition: model_fn is a string 

    Parameter out_name: desired folder name where outputs go
    Precondition: out_name is a string

    Returns: None (generates multiple csv files)
    """
    n1 = n1.reshape(n, m)
    n2 = n2.reshape(n, m)

    n1sum = n1.sum()
    n1percent = n1sum/(n*m)
    n1_lsum = num_conn_comps(n1.copy())
    n1_lpercent = n1_lsum/(n*m)

    newdirname = make_folder(dirname, model_fn, out_name)

    path1 = os.path.join(newdirname, 'predictions_rounded.csv')
    pd.DataFrame(n1).to_csv(path1, header=False, index=False)
    path2 = os.path.join(newdirname, 'predictions.csv')
    pd.DataFrame(n2).to_csv(path2, header=False, index=False)

    d = {'number total cells': [n1sum],
         'percent total cells': [n1percent],
         'number connected components': [n1_lsum],
         'percent connected components': [n1_lpercent]}
    path3 = os.path.join(newdirname, 'stats.csv')
    pd.DataFrame(data=d).to_csv(path3, index=False)
    return


def make_folder(dirname, model_fn, out_name):
    """
    If out_name is provided, generates a new folder with that name. Otherwise,
    generates a folder with the same name as model_fn.

    Parameter dirname: file path to folder containing images
    Precondition: dirname is a string

    Parameter model_fn: file name of model
    Precondition: model_fn is a string 

    Parameter out_name: desired folder name where outputs go
    Precondition: out_name is a string

    Returns: a string which is the new directory name to add images/files to
    """
    if out_name:
        newdirname = os.path.join(dirname, out_name)
    else:
        model_fn = os.path.normpath(model_fn).split(os.path.sep)[-1]
        newdirname = os.path.join(dirname, model_fn)
    if not os.path.exists(newdirname):
        os.makedirs(newdirname)
    return newdirname


def get_merged_img(dirname, img_files, img_name, n, m):
    """
    Merges the individual images in the n by m grid into one merged image
    and draws lines in between the individual images.

    Parameter dirname: file path to folder containing images
    Precondition: dirname is a string

    Parameter img_files: list of individual images of grid
    Precondition: img_files is List[string...]

    Parameter img_name: file name to give to merged image
    Precondition: img_name is a string

    Parameter n: number of rows of images
    Precondition: n is an int > 0

    Parameter m: number of cols of images
    Precondition: m is an int > 0

    Returns: None (generates new img file)
    """
    img_mat = np.zeros((n, m)).tolist()
    c1 = 0
    c2 = 0

    for f in img_files:
        if 'merged' in f:
            continue
        img_mat[c1][c2] = cv2.imread(f)
        c2 += 1
        if c2 == m:
            c2 = 0
            c1 += 1

    combined_img = cv2.vconcat([cv2.hconcat(list_h) for list_h in img_mat])
    draw_lines(combined_img)
    cv2.imwrite(os.path.join(dirname, img_name), combined_img)


def draw_lines(cv2_img):
    """
    Given an image, draws horizontal and vertical lines every 512 pixels.

    Parameter cv2_img: the merged image to draw lines on
    Precondition: cv2_img is a 3D numpy array [n*512,m*512,3]

    Returns: None (alters image)
    """
    y, x, _ = cv2_img.shape
    for i in range(x//512):
        if i == 0:
            continue
        cv2.line(cv2_img, (i*512, 0), (i*512, y), (255, 0, 0), thickness=2)
    for j in range(y//512):
        if j == 0:
            continue
        cv2.line(cv2_img, (0, j*512), (x, j*512), (255, 0, 0), thickness=2)


def np_to_csv(nparr, path, idx_to_name):
    """
    Generates csv value to path given np array 

    Parameter nparr: np array of data
    Precondition: np array of int or float of shape (n*m,7)

    Parameter path: folder path to write data to
    Precondition: path is a string

    Parameter idx_to_name: dictonary from class number to name
    Precondition: idx_to_name is Dict[int: string]

    Returns: None (generates csv)
    """
    df = pd.DataFrame(nparr, columns=idx_to_name.values())
    df.to_csv(path, index=False)


def np_class_dict_to_csv(d, path):
    """
    Generates csv value to path 

    Parameter d: dictionary of class name to corresponding np array
    Precondition: d is Dict[string: np array] where np arrays are of shape (n,m)

    Parameter path: folder path to write data to
    Precondition: path is a string

    Returns: None (generates csv)
    """
    with pd.ExcelWriter(path, engine='xlsxwriter') as writer:
        for k in d:
            df = pd.DataFrame(d[k])
            df.to_excel(writer, index=False, header=False, sheet_name=k)


def num_conn_comps(arr):
    """
    Given array of 1's and zeros, calculates the number of connected components.

    Parameter arr: array to search number of SCC's
    Precondition: arr is an np array with binary values

    Returns: number of connected components, an int
    """
    count = 0
    n, m = arr.shape
    for i in range(n):
        for j in range(m):
            if arr[i][j] == 1:
                dfs(arr, i, j, n, m)
                count += 1
    return count


def dfs(arr, i, j, n, m):
    """
    Perform recursive DFS

    Parameter arr: the array to peform DFS on
    Precondition: arr is an np array with binary values

    Parameter i: Row value of current cell
    Precondition: i is int between 0 and n

    Parameter j: Col value of current cell
    Precondition: j is int between 0 and m

    Parameter n: number of rows of images
    Precondition: n is an int > 0

    Parameter m: number of cols of images
    Precondition: m is an int > 0

    Returns: None
    """
    if i < 0 or j < 0 or i >= n or j >= m or arr[i][j] != 1:
        return
    arr[i][j] = 0
    dfs(arr, i+1, j, n, m)
    dfs(arr, i-1, j, n, m)
    dfs(arr, i, j+1, n, m)
    dfs(arr, i, j-1, n, m)


def parse_opt():
    """
    Parses command line output to make argparse.Namespace object

    Returns: argparse.Namespace object with parsed inputs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-lat', type=float, help='latitude', required=True)
    parser.add_argument('-lng', type=float, help='longitude', required=True)
    parser.add_argument('-n', type=int, help='num rows', required=True)
    parser.add_argument('-m', type=int, help='num cols', required=True)
    parser.add_argument('-z', type=int, help='zoom level (18 default)', const=18, default=18)
    parser.add_argument('-rm', type=str,
                        help='run multilabel model')
    parser.add_argument('-rb', type=str, help='run binary model')
    parser.add_argument('-ry', type=str, help='run yolo model')
    parser.add_argument(
        '-conf', type=float, help='confidence threshold for YOLO model (optional, defaults to 0.1)')
    parser.add_argument(
        '-outm', type=str, help='(optional) output folder name for multilabel model')
    parser.add_argument(
        '-outb', type=str, help='(optional) output folder name for binary model')
    opt = parser.parse_args()
    return opt


def main(lat, lng, n, m, z, rm, rb, ry, conf, outm, outb):
    """
    Scrapes images of latitude and longitude in n by m grid and runs given models
    """
    img_scrape_surr(lat, lng, n, m, z)
    dirname = os.path.join('outputs', f'{lat:.6f}_{lng:.6f}_{n:d}by{m:d}')
    if ry:
        run_yolo(dirname, n, m, ry, conf)
        return

    if rm:
        n1, n2 = run_xception(dirname, rm)
        run_multiclass(n1, n2, n, m, dirname, rm, outm)

    if rb:
        n1, n2 = run_xception(dirname, rb)
        run_binary(n1, n2, n, m, dirname, rb, outb)
    return


def run_xception(dirname, model_fn):
    """
    Runs binary/multilabel model 

    Parameter dirname: file path to folder containing images
    Precondition: dirname is a string

    Parameter model_fn: model file name
    Precondition: model_fn is a string

    Returns: n1, n2 two numpy arrays. n1 is np array containing binary values 
    of shape (n*m, num classes). n2 is np array containing float values of 
    shape (n*m, num classes)
    """
    dependencies = {
        'F1score': tfa.metrics.F1Score(num_classes=1, threshold=0.5, average='micro'),
        'precision': tf.keras.metrics.Precision(),
        'recall': tf.keras.metrics.Recall()
    }
    model = tf.keras.models.load_model(model_fn, custom_objects=dependencies)
    img_files = glob(os.path.join(dirname, '*.png'))
    l = []

    for f in img_files:
        if 'merged' in f:
            continue
        img = image.img_to_array(image.load_img(f))
        img = preprocess_input_xception(img)
        img = np.expand_dims(img, axis=0)
        l.append(model.predict(img))

    n2 = np.concatenate(l)
    n1 = np.rint(np.copy(n2))
    return n1, n2


if __name__ == "__main__":
    opt = parse_opt()
    main(**vars(opt))

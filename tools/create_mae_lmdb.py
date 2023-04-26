import os
import re

import cv2
import lmdb  # install lmdb by "pip install lmdb"
import numpy as np
import scipy.io as sio
import six
from PIL import Image
from tqdm import tqdm


def checkImageIsValid(imageBin):
  if imageBin is None:
    return False
  imageBuf = np.fromstring(imageBin, dtype=np.uint8)
  # imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
  if imageBuf.size == 0:
    return False
  img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
  imgH, imgW = img.shape[0], img.shape[1]
  if imgH * imgW == 0:
    return False
  return True


def writeCache(env, cache):
  with env.begin(write=True) as txn:
    for k, v in cache.items():
      txn.put(k.encode(), v)


def _is_difficult(word):
  assert isinstance(word, str)
  return not re.match('^[\w]+$', word)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
  """
  Create LMDB dataset for CRNN training.
  ARGS:
      outputPath    : LMDB output path
      imagePathList : list of image path
      labelList     : list of corresponding groundtruth texts
      lexiconList   : (optional) list of lexicon lists
      checkValid    : if true, check the validity of every image
  """
  if not os.path.exists(os.path.dirname(outputPath)):
    os.makedirs(os.path.dirname(outputPath))

  assert(len(imagePathList) == len(labelList))
  nSamples = len(imagePathList)
  env = lmdb.open(outputPath, map_size=1099511627776)
  cache = {}
  cnt = 0
  for i in tqdm(range(nSamples)):
    imagePath = imagePathList[i]
    labelPath = labelList[i]

    if not os.path.exists(imagePath) or not os.path.exists(labelPath):
      print('%s does not exist' % imagePath)
      continue
    
    with open(imagePath, 'rb') as f:
      imageBin = f.read()
    with open(labelPath, 'rb') as f:
      labelBin = f.read()
    
    if checkValid:
      if not checkImageIsValid(imageBin):
        print('%s is not a valid image' % imagePath)
        continue
      if not checkImageIsValid(labelBin):
        print('%s is not a valid image' % labelPath)
        continue

    imageKey = 'mosaic-%d' % cnt
    labelKey = 'gt-%d' % cnt
    cache[imageKey] = imageBin
    cache[labelKey] = labelBin
    if lexiconList:
      lexiconKey = 'lexicon-%09d' % cnt
      cache[lexiconKey] = ' '.join(lexiconList[i])
    if cnt % 1000 == 0:
      writeCache(env, cache)
      cache = {}
      print('Written %d / %d' % (cnt, nSamples))
    cnt += 1
  nSamples = cnt
  cache['num-samples'] = str(nSamples).encode()
  writeCache(env, cache)
  print('Created dataset with %d samples' % nSamples)

if __name__ == "__main__":
  
  output_dir = '/home/mrchen/cmr/mosaic/DiG/npy_dir'
  image_path_list, label_list = [], []
  # masked_image_path_list, masked_label_list = [], []
  
  gt_dir = '/home/mrchen/cmr/mosaic/KAIR/sr3/datasets/english_easy_train/hr_(128,512)'
  mosaic_dir = '/home/mrchen/cmr/mosaic/KAIR/sr3/datasets/english_easy_train/lr_(32,128)'
  for filename in os.listdir(gt_dir):
    if filename.endswith('.jpg'):
        label_list.append(os.path.join(gt_dir, filename))
        image_path_list.append(os.path.join(mosaic_dir, filename))

  image_dir = '/home/mrchen/cmr/mosaic/DiG/npy_dir'

  lmdb_output_path = os.path.join(output_dir, 'gt_lmdbs')
  # masked_lmdb_output_path = os.path.join(output_dir, 'mosaic_lmdbs')

  createDataset(lmdb_output_path, image_path_list, label_list)

#!/usr/bin/env python
"""
  Class SpecImage that holds info for a set of spectra (usually from an image file)
  Assume image file has format "x y i1 i2 ... iN" (that is, one whole pixel spectrum per line) and first line has wavelengths
  It also assumes that wavelengths are ordered (increasing or decreasing, but not random)

""" 
import matplotlib
matplotlib.use('Agg') # do not use the X11 as default, e.g. on HPC (must be before any other module)
import gzip 
import pickle
import re         # regular expressions, perl-style
import os, re, gzip, string, sys
import math
import numpy as np
import statsmodels.api as sm
from matplotlib import colors 
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy import interpolate, ndimage, signal, sparse, spatial, optimize, stats
from sklearn import manifold, metrics, cluster, neighbors, decomposition, preprocessing
from collections import Counter

global_vars = [True] ## used by multiprocessing.Pool

class SpecImage:
  """
    * wl is a 1D array with the wave numbers (in cm-1)
    * spc is a 2D array with the spectra. Size may differ from size of XY since spc stores only valid, and distinct spectra
    * xy are the _original_ image coordinates (that is, before subsampling, removal, etc.)
    * xy_null is a list of coordinates for all removed/invalid spectra
    * idx is a mapping between XY and SPC, with same size as XY (since several pixels may point to same spectrum).
  """
  def __init__(self, copyfrom=None, filename=None):
    self.xy_null = None
    if filename:
      import h5py
      mlab=h5py.File(filename,"r")
      self.wl = np.array(mlab["wl"],dtype=float)
      self.xy = np.array(mlab["xy"],dtype=float)
      self.spc = np.array(mlab["spc"],dtype=float)
      self.idx = np.array(mlab["idx"],dtype=int)
      self.description = str(mlab["description"])
      if "xy_null" in mlab:
        self.xy_null = np.array(mlab["xy_null"],dtype=int)
      mlab.close()
    else:
      if copyfrom is None:
        self.wl = np.array([],dtype=float)
        self.xy = np.array([],dtype=float).reshape(-1,2)
        self.spc = np.array([[]],dtype=float)
        self.idx = np.array([],dtype=int) # ID wrt the original file
        self.description = ""
      else:
        self.wl = np.array(copyfrom.wl, copy=True)
        self.xy = np.array(copyfrom.xy, copy=True)
        self.spc = np.array(copyfrom.spc, copy=True)
        self.idx = np.array(copyfrom.idx, copy=True)
        self.description = copyfrom.description
        try:
          self.xy_null = copyfrom.xy_null
        except AttributeError:
          # print "SpecImage object is from old module version, which does not have xy_null parameter. Which is fine by the way."
          pass
        self.remove_nan_pixels()

  def save(self, filename=None):
    import h5py
    if filename is None:
      filename = "spc" + self.description.strip() + ".mat"
    mlab=h5py.File(filename,"w")
    mlab.create_dataset("wl", data=self.wl, compression="gzip")  
    mlab.create_dataset("xy", data=self.xy, compression="gzip")
    mlab.create_dataset("spc", data=self.spc, compression="gzip")
    mlab.create_dataset("idx", data=self.idx, compression="gzip")
    mlab.create_dataset("description", data=self.description) # scalars can't be compressed
    try:   ## if self.xy_null is not None:  
      ##  http://stackoverflow.com/questions/610883/how-to-know-if-an-object-has-an-attribute-in-python
      mlab.create_dataset("xy_null", data=self.xy_null, compression="gzip")
    except:
      pass
    mlab.close()

  def read_from_matlab(self, mfilename=None, varname=None):
    if (mfilename is None):
      print ("I need the name of matlab file\n")
      return
    import scipy.io
    mlab=scipy.io.loadmat(mfilename)
    if (varname is None):
      varname = "Data_rr"  # guess
    varname = str(varname)
    img=mlab[varname]
    if (len(img.shape) == 2): # no dimensional info; must guess (closest to a square)
      n_x = int(factorise(img.shape[0])[0]) # first element is closest to srqt(n)
      n_y = int(img.shape[0]/n_x)
    else: # img.shape == 3 dimensions 
      n_x = img.shape[1]
      n_y = img.shape[0]
    #EXAMPLE# img = np.transpose(img,(1,0,2)) # invert X and Y leaving wl untouched
    self.xy = np.array([[n_x-1-x,y] for y in range(n_y) for x in range(n_x)], dtype=float)
    self.wl = np.array(mlab["xaxis"][:,0], dtype=float) # it's an Nx1 matrix, we want the "first" (and only) column 
    self.spc = np.array(img,dtype=float).reshape(n_x * n_y, -1) # minus one means to calculate; ammounts to the number of wavelengths
    self.idx = np.arange(0, n_x * n_y)
    if self.wl[0] > self.wl[-1]:
      self.wl = self.wl[::-1]  #same as self.wl.reverse()
      self.spc = np.fliplr(self.spc)  # flipr() == [::-1] == inverts the order left-right
    self.remove_nan_pixels()
    self.description=varname
    return self
  
  def read_witec(self, filename=None, excitation_wavelength=None, is_zipped=None):
    """
      Import WiTec text files. Three files must be present, with suffixes "Header", "X-Axis" and "Y-axis". The Y-Axis
      file may be zipped, since tends to be very large
    """
    if is_zipped is None:
      zipped = False
    if filename is None:
      print ("I need the (single) common prefix for the 3 filenames (before the \"(Header).txt\" etc.)\n")
      return
    
    for line in open (filename + "(Header).txt", "r"):
      if "SizeGraph" in line: # redundant: "X-Axis" file will also give this info
        size_graph = int(line.split(" ")[-1].strip())      # store last column using string.split()
      if "SizeX" in line:
        n_x = int(line.split(" ")[-1].strip())      # store last column using string.split()
      if "SizeY" in line:
        n_y = int(line.split(" ")[-1].strip())      # store last column using string.split()
    self.wl = np.array([float(line) for line in open (filename + "(X-Axis).txt", "r")], dtype=float) # assumes ONE column file 

    if excitation_wavelength:
      self.wl = (1./float(excitation_wavelength) - 1./self.wl) * 10**7 # transform wavelengths (nm) into wave numbers (cm-1)
    if size_graph != self.wl.shape[0]:
      print ("Wave length 'SizeGraph' described by Header (= " + size_graph + ") disagrees with X-Axis file (= " + \
        self.wl.shape[0] + ")\n")
    if is_zipped:
      # minus one on reshape -> divide total size by new size (will lead to wl size)
      self.spc = np.array([float(line) for line in gzip.open (filename + "(Y-Axis).txt.gz", "r")], dtype=float).reshape(n_x * n_y, -1) 
    else:
      self.spc = np.array([float(line) for line in open (filename + "(Y-Axis).txt", "r")], dtype=float).reshape(n_x * n_y, -1) 
    self.idx = np.arange(0, n_x * n_y)
    self.xy = np.array([[x,y] for y in range(n_y) for x in range(n_x)], dtype=float)
    self.description = filename
    return self  

  def read_image(self, filename, is_zipped=None, remove_zero_wl=None):
    if is_zipped is None:
      is_zipped=True
    if is_zipped:
      ifile = gzip.open(filename, "r")
    else:
      ifile = open(filename, "r")
    if remove_zero_wl is None:
      remove_zero_wl = True
    is_reversed = False
    self.description = str(filename)
    line = ifile.readline().rstrip("\n\t ") # first line are wavelengths
    self.wl = np.array([float(i) for i in re.split("\s+",line.lstrip()) if i])
    # check if wave numbers are in crescent order
    if self.wl[0] > self.wl[-1]:
      is_reversed = True
      self.wl = self.wl[::-1]  #same as self.wl.reverse()
    i_idx = 2; f_idx = len(self.wl) + 2  # plus two since dlines[] will also have XY info
    if remove_zero_wl:# find smallest index larger than "zero" (~10) wave number to define slice
      zero_idx = next(i for i,j in enumerate(self.wl) if j > 10)
      self.wl = self.wl[zero_idx:]
      if is_reversed:
        f_idx -= zero_idx 
      else:
        i_idx += zero_idx 
    # now read XY and SPC values
    line = ifile.readline().rstrip("\n\t ")
    dlines = []
    while line: # first read whole file in memory as list (much faster than updating spc) 
      dlines.append([float(i) for i in re.split("\s+",line.lstrip().rstrip())])
      line = ifile.readline().rstrip("\n\t ")
      if not len(dlines)%1000:
        sys.stdout.write("."); sys.stdout.flush()
    sys.stdout.write("*"); sys.stdout.flush()
    ifile.close()
    self.xy  = np.array([x[:2] for x in dlines], dtype=float) # first elements are X,Y 
    self.spc = np.array([x[i_idx:f_idx] for x in dlines], dtype=float) # other elements are intensities at this location
    
    self.idx = np.arange(len(self.xy)) 
    if is_reversed:
      self.spc = np.fliplr(self.spc)  # flipr() == [::-1] == inverts the order left-right
    self.remove_nan_pixels()
    sys.stdout.write("+\n");
    return self

  def change_resolution(self, resolution=None):
    """
    """
    if resolution is None:
      resolution = 200
    xy=np.array(xy, dtype=float)
    if xy_null is not None:
      xy=np.concatenate((xy, xy_null),0)
      if null_zvalue is None:
        null_zvalue = z.max() + 0.1 * np.std(z)
      z =np.concatenate((z,np.repeat(null_zvalue, xy_null.shape[0])),0)
    # from http://stackoverflow.com/questions/15586919/inputs-x-and-y-must-be-1d-or-2d-error-in-matplotlib
    rangeX = max(xy[:,0]) - min(xy[:,0])
    rangeY = max(xy[:,1]) - min(xy[:,1])
    rmax = max(rangeX,rangeY)
    rangeX /= rmax
    rangeY /= rmax 
    xi = np.linspace(min(xy[:,0]), max(xy[:,0]), int(resolution * rangeX))
    yi = np.linspace(min(xy[:,1]), max(xy[:,1]), int(resolution * rangeY))
    X, Y = np.meshgrid(xi, yi)
    Z = interpolate.griddata((xy[:,0], xy[:,1]), np.array(z), (X, Y), method='nearest')
    extent=[min(xy[:,0]),max(xy[:,0]),min(xy[:,1]),max(xy[:,1])]
    return Z, extent

    
  def set_valid_pixels(self, valid_pixels, are_invalid = None):
    """
      reduces self.spc to only those elements in valid_pixels[]; IDX is updated with new indexes, and temporary values "-1"
      indicate that XY and IDX should be updated (removed). valid_pixels[] can also be the list of invalid indexes, if
      the boolean are_invalid is set.
    """
    if are_invalid is True:
      valid_pixels = np.array(list( set(range(self.spc.shape[0])) - set(valid_pixels) )) # complement to invalid list

    valid_pixels = np.array(valid_pixels)
    if (valid_pixels.shape[0] == self.spc.shape[0]):
        return ## nothing to do
    valid_pix = [int(i) for i in sorted(set(valid_pixels))] # although set had sorted elements already
    self.spc = np.array([self.spc[i] for i in valid_pix], dtype=float)
    valid_dict = {j:i for i,j in enumerate(valid_pix)}
    valid = []
    invalid = [] 
    for i in range(len(self.idx)):
      try:
        self.idx[i] = valid_dict[self.idx[i]] # new value is simply index in spc[] (same location as valid[])
        valid.append(i)
      except KeyError:
        self.idx[i] = -1
        invalid.append(i)
    try:
      self.xy_null = np.concatenate((self.xy_null,self.xy[invalid]),0)
    except AttributeError:
      self.xy_null = np.array(self.xy[invalid],dtype=float)
    except ValueError:
      self.xy_null = np.array(self.xy[invalid],dtype=float)
    self.idx = self.idx[valid] 
    self.xy  = self.xy[valid]
    return self

  def quick_remove_pixels_list(self, invalid = None):
    """
    Quick-and-dirty solution to remove pixels before anything else is done. Work-around for cosmic spike removal, for
    instance.
    OBS: Does NOT work if index list is used (that is, any mapping/simplification was applied), since assumes XY and IDX
    lists have same size (number of pixels in SPC.)
    """
    if invalid is None:
      return self
    valid_pixels = np.array(list( set(range(self.spc.shape[0])) - set(invalid) )) # complement to invalid list
    self.spc = self.spc[valid_pixels]
    ##self.idx = self.idx[valid_pixels]
    self.idx = np.arange(valid_pixels.shape[0])
    self.xy  = self.xy[valid_pixels]
    return self

  def remove_nan_pixels(self):
    invalid_bool = [math.isnan(np.mean(spec)) for spec in self.spc]
    if not any(invalid_bool):
      return self
    self.set_valid_pixels([i for i in range(len(invalid_bool)) if not invalid_bool[i]])
    return self

  def remove_negative_valued_pixels(self, tolerance=None):
    if tolerance is None:
      tolerance = 0.01
    tolerance = int(tolerance * self.spc.shape[1]) # how many wave numbers
    invalid_bool = [(np.sum(spec<0) > tolerance) for spec in self.spc]
    if not any(invalid_bool):
      return self
    self.set_valid_pixels([i for i in range(len(invalid_bool)) if not invalid_bool[i]])
    return self

  def remove_noisy_pixels(self, valid_regions=None, debug=None, find_only=None):
    """
      Remove spectra that look like white noise, based on the p-values of the autocorrelation function (ACF). Also
      remove spectra that are too similar to a constant value (since the ACF cannot handle constant series.)
    """
    if valid_regions is None:
      valid_regions = [[100,3000]] # 1800~2700 is silent region but we want whole series (apart from extremes)
    valid = [] # notice that include[] is a list, not an np.array
    for vreg in valid_regions:
      valid += [i for i in range(len(self.wl)) if (self.wl[i] >= vreg[0] and self.wl[i] <= vreg[1])]
    valid_pixels = []
    invalid = []
    for i in range(self.spc.shape[0]):
      # baseline correction using 20% of points
      spec = ndimage.morphology.white_tophat(self.spc[i][valid], None,  np.repeat([1], int(self.wl.size * 0.2) + 1))
      spec = spec - np.min(spec)
      if spec.sum() > 1e-4: # do not run ACF if all zeros
        _, _, pvalue = sm.tsa.stattools.acf(spec,nlags=20,qstat=True) # 3 arrays: acf, qstats and pvalue 
        sum_pvalue = sum(pvalue[:20])
      else:
        sum_pvalue = 1
      if sum_pvalue < 0.1:
        valid_pixels.append(i)
      else:
        invalid.append(i)
    if debug and len(invalid):
      sys.stdout.write("DEBUG: list of noise pixels: " + str(invalid) + "\n")
    if find_only is not True:
        self.set_valid_pixels(valid_pixels=valid_pixels)
        return self
    else: # just return a list of valid and invalid indexes
        return valid_pixels, invalid
  
  def remove_cosmic_spikes(self, sd_threshold=None, n_iter=None, debug=None, neighbours=None, mode=None):
    """
      Find cosmic spikes based on standardized derivatives over pixels per wavenumber, and over wavenumbers of each
      spectrum. sd_threshold is the number of standard deviations away (from a N(0,1)) to detect a peak. n_iter controls
      how many iterations should it look for peaks, shuffling the pixels location and reducing SD at each iteration.
      Based on idea from doi:10.1366/12-06660 (Mozharov et al. Appl Spectro 2012, p.1326), with important differences:
      (1) z-score computation to find outliers, removing conditional tests (since spikes _can_ leak to neighboring pixels)
      (2) complete hyperspec scan over wavenumbers and pixels
      (3) spike removal by baseline-type interpolation, assuming closest neighbors 
      (4) iterative permutation over pixels

      "debug" produces a longer output, with [iteration, pixel, wl] list for, resp: scan over WL, scan over PX, true
      peaks (found in both) and false peaks (found in only one of scans)
    """
    if mode is None:
      mode = 0
    if neighbours is None:
      neighbours = 1 
    if neighbours> 10:
      neighbours = 10 
    if sd_threshold is None or sd_threshold < 2: # S.D. threshold for assigning a spike
      sd_threshold = 6
    if n_iter is None or n_iter < 1:
      n_iter = 2
    sd = np.linspace (2 * sd_threshold, sd_threshold, n_iter)
    if (n_iter < 2):
      sd = [sd_threshold] # only one iteration
    no_px =  list(range(self.spc.shape[0])) # reordering of pixels (spectra)
    no_wl =  list(range(self.spc.shape[1])) # reordering of wavenumbers
    if debug:
      truepeak = np.array([[]],dtype=int).reshape(0,3)
      falspeak = np.array([[]],dtype=int).reshape(0,3)
      wl_peak  = np.array([[]],dtype=int).reshape(0,3)
      px_peak  = np.array([[]],dtype=int).reshape(0,3)
      
    for iteration in range(n_iter):
      np.random.shuffle(no_px) # in-place reordering
      pixel_wl_1 = np.array([], dtype=int).reshape(0,2)
      pixel_wl_2 = np.array([], dtype=int).reshape(0,2)
      for wl in no_wl:
        dif = np.diff(self.spc[no_px,wl]) # vector of pixels, in random order, for this wavenumber wl
        dif = (dif - np.mean(dif))/np.std(dif)
        list_of_pixels = np.where(dif < - sd[iteration])[0] # list of indexes with extreme values (potentials spike)
        # create a pair [pixel, wl] for each peak at this wavenumber and add to overall list
        if list_of_pixels.shape[0]:
          pixel_wl_1 = np.concatenate((pixel_wl_1, np.array([[no_px[i],wl] for i in list_of_pixels])))
      for px in no_px: # it can be in natural order
        # remove obvious trends, and possibly use random order of wavenumbers, before differentiating
        dif = np.diff(ndimage.morphology.white_tophat(self.spc[px,no_wl], None, np.repeat([1], int(self.wl.size * 0.1) + 1)))
        dif = (dif - np.mean(dif))/np.std(dif)
        list_of_wl = np.where(dif < - sd[iteration])[0] # list of wavenumbers with extreme values (potentials spike)
        # create a pair [pixel, wl] for each peak from this spectrum and add to overall list
        if list_of_wl.shape[0]:
          pixel_wl_2 = np.concatenate((pixel_wl_2, np.array([[px,no_wl[i]] for i in list_of_wl])))
      
      # http://stackoverflow.com/questions/9269681/intersection-of-2d-numpy-ndarrays 
      p1_view = pixel_wl_1.view([('', int)] * pixel_wl_1.shape[1]) # pairs [pixel,wl] become tuples
      p2_view = pixel_wl_2.view([('', int)] * pixel_wl_2.shape[1]) # btw, we know that pixel_wl_2.shape[1] is 2
      if mode == 0:  # peaks are only those found by both methods
        pixel_wl = np.intersect1d(p1_view, p2_view)
      elif mode == 1: # peaks are only those found for a given wl across pixels
        pixel_wl = p1_view 
      elif mode == 2: # peaks are only those found for a given spectrum across wavenumbers
        pixel_wl = p2_view 
      else: # peaks found by any method will be replaced
        pixel_wl = np.union1d(p1_view, p2_view)
      pixel_wl = pixel_wl.view(int).reshape(-1, pixel_wl_1.shape[1])

      if debug: # add column with iter number to pixel_wl[], and then concatenate to big table
        #  table with all peaks found scanning over WL and pixels 
        thistab = np.hstack((np.repeat(iteration,pixel_wl_1.shape[0]).reshape(pixel_wl_1.shape[0],-1),pixel_wl_1))
        wl_peak = np.concatenate((wl_peak, thistab), axis=0)
        thistab = np.hstack((np.repeat(iteration,pixel_wl_2.shape[0]).reshape(pixel_wl_2.shape[0],-1),pixel_wl_2))
        px_peak = np.concatenate((px_peak, thistab), axis=0)
        #  table with only peaks found by both ways
        thistab  = np.hstack((np.repeat(iteration,pixel_wl.shape[0]).reshape(pixel_wl.shape[0],-1),pixel_wl))
        truepeak = np.concatenate((truepeak, thistab), axis=0)
        # we also store peaks that turned out to be false since appeared in only one 
        false_wl = np.setxor1d(p1_view, p2_view)  # points that were found in only one of the scans
        false_wl = false_wl.view(int).reshape(-1, pixel_wl_1.shape[1])
        thistab  = np.hstack((np.repeat(iteration,false_wl.shape[0]).reshape(false_wl.shape[0],-1),false_wl))
        falspeak = np.concatenate((falspeak, thistab), axis=0)
      
      for px in set(pixel_wl[:,0]):  ## pixels where there is a spike
        peak_list = np.array(sorted(set(pixel_wl[np.where(pixel_wl[:,0] == px)[0],1])))
        for i in range(neighbours):
          peak_list = np.concatenate((peak_list, peak_list - 1, peak_list + 1)) # more neighbors
        peak_list = np.array(sorted(set(peak_list)))
        peak_list = peak_list[ np.where(peak_list >= 0)[0] ] # remove negative values
        peak_list = peak_list[ np.where(peak_list < self.wl.shape[0])[0] ] # remove values exceeding largest WL
        dif = (self.spc[px] - np.mean(self.spc[px]))/np.std(self.spc[px])
        exclude_interpol = np.concatenate((peak_list, np.where(dif > 8)[0])) # remove all extreme values from interpolation
        valid = np.array(list( set(range(self.wl.shape[0])) - set(exclude_interpol) ))  # array, not list (safer for slices)
        if valid.shape[0] > 2 * peak_list.shape[0]: # whole spectrum can be outlier if intensity larger than neighbours
          #interpol = interpolate.InterpolatedUnivariateSpline(self.wl[valid], self.spc[px,valid]) # spline OO function
          #interpol = interpolate.interp1d(self.wl[valid], self.spc[px,valid], kind="cubic") # cubic OO function
          #self.spc[px,peak_list] = interpol(self.wl[peak_list]) # replace peaks by interpolated values
          self.spc[px] = np.interp(self.wl, self.wl[valid], self.spc[px,valid]) ## much much faster than OO functions
      
    if debug:
      return self, wl_peak, px_peak, truepeak, falspeak 
    else:
      return self
  
  def remove_cosmic_spikes_rescaled(self, sd_threshold=None, n_iter=None, neighbours=None):
    """
      Find cosmic spikes based on standardized derivatives over pixels per wavenumber, and over wavenumbers of each
      spectrum. sd_threshold is the number of standard deviations away (from a N(0,1)) to detect a peak. n_iter controls
      how many iterations should it look for peaks, shuffling the pixels location and reducing SD at each iteration.
      Based on idea from doi:10.1366/12-06660 (Mozharov et al. Appl Spectro 2012, p.1326), with important differences:
      (1) z-score computation to find outliers, removing conditional tests (since spikes _can_ leak to neighboring pixels)
      (2) complete hyperspec scan over wavenumbers and pixels
      (3) spike removal by baseline-type interpolation, assuming closest neighbors 
      (4) iterative permutation over pixels

      Alternative where reshuffling takes place first, and all spectra are rescaled to the z-score
    """
    if neighbours is None:
      neighbours = 1 
    if neighbours> 10:
      neighbours = 10 
    if sd_threshold is None or sd_threshold < 2: # S.D. threshold for assigning a spike
      sd_threshold = 4
    if n_iter is None or n_iter < 1:
      n_iter = 5
    no_px =  range(self.spc.shape[0]) # reordering of pixels (spectra)
    
    meanval  = np.array([np.mean(spec) for spec in self.spc])
    sdval    = np.array([np.std(spec) for spec in self.spc])
    self.spc = np.array([(spec - mu)/sd  for spec,mu,sd in zip(self.spc,meanval,sdval)],dtype=float)

    pixel_wl_1 = np.array([], dtype=int).reshape(0,2)
    for iteration in range(n_iter):
      np.random.shuffle(no_px) # in-place reordering
      for wl in range(self.spc.shape[1]):
        dif = np.diff(self.spc[no_px,wl]) # vector of pixels, in random order, for this wavenumber wl
        dif = (dif - np.mean(dif))/np.std(dif)
        list_of_pixels = np.where(dif < - sd_threshold)[0] # list of indexes with extreme values (potentials spike)
        # create a pair [pixel, wl] for each peak at this wavenumber and add to overall list
        if list_of_pixels.shape[0]:
          pixel_wl_1 = np.concatenate((pixel_wl_1, np.array([[no_px[i],wl] for i in list_of_pixels])))
    # array -> tuples followed by Counter().items() for frequencies
    pixel_wl = np.array([i for i,j in Counter(tuple([tuple(i) for i in pixel_wl_1])).items() if (j>n_iter/2)],dtype=int)

    for px in no_px:
      peak_list = np.array(sorted(set(pixel_wl[np.where(pixel_wl[:,0] == px)[0],1])), dtype=int)
      for i in range(neighbours):
        peak_list = np.concatenate((peak_list, peak_list - 1, peak_list + 1)) # remove neighbors
      peak_list = np.array(sorted(set(peak_list)),dtype=int)
      peak_list = peak_list[ np.where(peak_list >= 0)[0] ] # remove negative values
      peak_list = peak_list[ np.where(peak_list < self.wl.shape[0])[0] ] # remove values exceeding largest WL

      dif = np.diff(ndimage.morphology.white_tophat(self.spc[px], None, np.repeat([1], int(self.wl.size * 0.2) + 1)))
      dif = (dif - np.mean(dif))/np.std(dif)
      peak_list = np.concatenate((peak_list, np.where(dif > 2 * sd_threshold)[0])) # remove all extreme values from interpolation
      valid = np.array(list( set(range(self.wl.shape[0])) - set(peak_list) ),dtype=int)  # array, not list (safer for slices)
      if valid.shape[0] > 2 * peak_list.shape[0]: # whole spectrum can be outlier if intensity larger than neighbours
        interpol = interpolate.InterpolatedUnivariateSpline(self.wl[valid], self.spc[px,valid], k=5) # spline OO function
        for i in peak_list: # replace peak by interpolated value
          self.spc[px,i] = interpol(self.wl[i])
#          self.spc[px,i] = -20
      
    # revert spectra to original scale
    self.spc = np.array([spec * sd + mu  for spec,mu,sd in zip(self.spc,meanval,sdval)],dtype=float)
    return self

  def set_valid_area(self, x=None, y=None):
    if x is None:
      x = False
    if y is None:
      y = False
    valid_xy = [i for i in range(self.xy.shape[0]) if 
        (((not x) or (x[0] <= self.xy[i,0] <= x[1])) and ((not y) or (y[0] <= self.xy[i,1] <= y[1])))]
    # valid_xy is indexed by xy (or idx) arrays; must be mapped to spc array (some repetitions maybe):
    self.set_valid_pixels(self.idx[valid_xy]) # also removes elements from XY (and IDX) outside area (all map to pixel "-1")
    return self

  def flip_xy(self):
    """ 
      change X and Y information (for when we mistakenly read the transpose of the image, mixing Y and Y dimensions)
    """
    self.xy = np.array([[x,y] for x in sorted(set(self.xy[:,1])) for y in sorted(set(self.xy[:,0]))])
        
  def set_subsample(self, sample=None):
    if sample is None or sample >= 1.:
      sample = 0.1
    valid = np.random.choice(self.spc.shape[0], int(sample * self.spc.shape[0]), replace=False)
    self.set_valid_pixels(valid)
    return self

  def set_wavelength_interval(self, lower, upper):
    valid=[i for i in range(len(self.wl)) if (self.wl[i] > lower) and (self.wl[i] < upper)]
    self.wl = self.wl[valid]
    self.spc = self.spc[:,valid]
    return self

  def set_wavelength_intervals(self, valid_regions=None):
    """ 
      Matrix of regions (in wave number units) to be maintained. Others (silent region, extremes) are excluded.
      Notice that this is a newer version of the set_wavelength_interval(), which allowed only extremes...
    """
    if valid_regions is None:
      valid_regions = [[100,1900],[2600,3200]] # 1800~2700 is silent region, but we can keep a few values
    valid = [] # notice that include[] is a list, not an np.array
    for vreg in valid_regions:
      valid += [i for i in range(len(self.wl)) if (self.wl[i] >= vreg[0] and self.wl[i] <= vreg[1])]
    self.wl = self.wl[valid]
    self.spc = self.spc[:,valid]
    return self

  def interpolate_spline(self, frac=None, spline=None, interval=None):
    if frac==None or frac<0.01:
      frac=1
    if spline is None:
      spline=False
    if interval is None:
      interval = [self.wl.min(),self.wl.max()]
    if interval[0] < self.wl.min():
      interval[0] = self.wl.min()
    if interval[1] > self.wl.max():
      interval[1] = self.wl.max()
    new_wl=np.linspace(interval[0],interval[1],int(frac * len(self.wl)))
    if spline:
      tck=[interpolate.splrep(self.wl,row) for row in self.spc] ## it's linear iff s=0 explicitly
      self.spc=np.array([interpolate.splev(new_wl,i,der=0) for i in tck],dtype=float) 
    else:
      self.spc=np.array([np.interp(new_wl,self.wl, row) for row in self.spc],dtype=float)
    self.wl=new_wl
    return self

  def interpolate_savgol(self, n=None, order=None, deriv=None):
    """
      Interpolation followed by Savitzky-Golay filter. Notice that the SavGol filter must be applied on equally-spaced
      points, and therefore this function may change the wavelengths. If further simplification (subsampling over
      wavelengths) is needed, then run interpolation_spline() *after* applying the SavGol filter. "n" and "order" are the parameters 
      for the SavGol slidding window, while "deriv" is the order of the derivative (zero meaning no derivative calculation) 
    """
    if n is None:
      n = 5
    if not n%2:
      n += 1
    if order is None:
      order = (n+2)/2
    order = int(order)
    if deriv is None or deriv is False:
      deriv=0
    if deriv is True or deriv > order: # deriv should be a number, but user might mistakenly use it as bool
      deriv=2
    new_wl=np.linspace(min(self.wl),max(self.wl),len(self.wl))
    self.spc=np.array([signal.savgol_filter(np.interp(new_wl,self.wl, row), window_length=n, polyorder=order, deriv=deriv) \
      for row in self.spc],dtype=float)
    self.wl=new_wl
    return self

  def moving_average_smoothing(self, n=None, power=None):
    """
      Smooths all spectra as well as wavelength using a vector of powered triangular weights in a moving average
      (slidding window). A power lower than one  generates a flat (uniform) weight for all 2n-1 points, while higher
      values give much more weight to central  points. 
      Notice that this function will change the wave number values and even vector size (since they must be smoothed as well)
    """
    if n is None or n < 2:
      n = 3
    if power is None:
      power = 2.
    n = int(n)
    weights = np.power( np.convolve (np.repeat(1./float(n), n), np.repeat(1./float(n), n)), power ) # exponentiated triangular weights
    weights /= sum(weights)
    self.wl = moving_average(self.wl, weights=weights)
    self.spc=np.array([moving_average(row, weights=weights) for row in self.spc],dtype=float)
    return self

  def lowess(self, frac=None):
    if frac==None or frac>0.9:
      frac=0.01
    self.spc=np.array([sm.nonparametric.lowess(row, self.wl, frac=frac, is_sorted=True, return_sorted=False) for row in self.spc],dtype=float)
    return self

  def rescale_01(self):
#   for spec in np.nditer(self.spc, op_flags=['readwrite']): # overwrite values as we iterate --> does not work, goes element-wise
    self.spc = np.array([rescale_spectrum_01(spec) for spec in self.spc],dtype=float)
    return self
  
  def rescale_zero(self):
    self.spc = np.array([spec - min(spec) for spec in self.spc],dtype=float)
    return self
  
  def rescale_sum(self):
    self.spc = np.array([spec/np.sum(spec) for spec in self.spc],dtype=float)
    return self
  
  def rescale_mean(self):
    self.spc = np.array([spec/np.mean(spec) for spec in self.spc],dtype=float)
    return self
  
  def rescale_zscore(self):
    self.spc = np.array([(spec - np.mean(spec))/np.std(spec) for spec in self.spc],dtype=float)
    return self
 
  def background_correction_medpolish(self, max_iter=None): # not actually a "background correction" 
    """
      Replaces the signals by the residuals of a two-way table with an additive model, which can be found
      through the Median polish algorithm. The rows and columns correspond to pixels and wave numbers, respectively. The
      additive model is given by y_ij = x + a_i + b_j + e_ij (the median polish maintains all these values at all
      iterations).
    """
    if max_iter is None:
      max_iter = 2
    self.spc, pixel_effect, wl_effect = median_polish(self.spc, max_iter=max_iter)
    return self, pixel_effect, wl_effect


  def simplify_spectral_matrix(self, npoints=None, n_neighbors=None, smooth_fraction=None, smooth_size=None, valid_regions=None):
    """
      removes noisy spectra and merge remaining similar ones. 
      Also creates updated list of IDX: an index mapping pixels before and after simplification, which may be used as clustering label. 
      All spectra are baseline-corrected (tophat) for analysis, and the indexing/classification is based on a rough
      interpolation. Furthermore only non-noise spectra are maintained.
      If "n_neighbors" is an integer greater than zero, than clustering is constrained to pixels at this vicinity.
    """
    self.remove_noisy_pixels(valid_regions=valid_regions)
    if npoints is None:
      npoints = 512
    if npoints >= self.spc.shape[0]:
      return # more clusters than objects
    if n_neighbors and n_neighbors > 0.5 * self.spc.shape[0]:
      n_neighbors = 0.5 * self.spc.shape[0]
    if n_neighbors and n_neighbors < 3:
      n_neighbors = 3
    if smooth_fraction is None:
      smooth_fraction = 0.2
    if smooth_size is None:
      smooth_size=10
    if valid_regions is None:
      valid_regions = [[100,1900],[2600,3000]] # 1800~2700 is silent region, but we can keep a few values
    
    simple = SpecImage() # temporary specs therefore don't need wl, idx, or description
    simple.xy = np.array(self.xy) # copy
    simple.idx = np.repeat(-1, self.idx.shape[0]) # index with same size as *original* (not simplified) SpecImage 
    valid_idx = [] # indexes from *original* specs that should be maintained after simplification; others are purged
    location_map = [] # maps idx from simplified to original spc
    sm_weights = np.repeat(1./float(smooth_size), smooth_size) # uniform weights; default is triangular
    smooth_wl = moving_average(self.wl, weights=sm_weights) # used for rough spectra (quick clustering)
    rough_wl = np.linspace(smooth_wl[0], smooth_wl[-1], int(smooth_fraction * len(smooth_wl)))
    if valid_regions:
      include_rough=[]
      for vreg in valid_regions:
        include_rough += [i for i in range(len(rough_wl)) if (rough_wl[i] >= vreg[0] and rough_wl[i] <= vreg[1])]
      rough_wl = np.sort([rough_wl[i] for i in include_rough])

    for i in range(self.spc.shape[0]):
      corrected_spc = ndimage.morphology.white_tophat(self.spc[i], None,  np.repeat([1], int(self.wl.size * 0.2) + 1))
      spectrum = moving_average(corrected_spc, weights=sm_weights) # spectrum points are associated to smooth_wl values 
      rough_spec = np.interp(rough_wl, smooth_wl, spectrum) # some regions are excluded 
      if simple.spc.size:
        simple.spc = np.vstack([simple.spc, rough_spec])
      else:
        simple.spc = np.array(rough_spec, dtype=float)
      simple.idx[np.where(self.idx == i)[0]] = i # simple.idx has original size but points to elements in new spc[]
      location_map.append(i)

    valid=[i for i,j in enumerate(simple.idx) if j > -1] # remove XY/IDX info about excluded spectra
    simple.idx = simple.idx[valid] 
    simple.xy  = simple.xy[valid]
    if n_neighbors:
      connect = kneighbors_graph_by_idx (simple.xy, simple.idx, n_neighbors=n_neighbors)
      kmod=cluster.AgglomerativeClustering(n_clusters=npoints,connectivity=connect,affinity="cosine",linkage="average").fit(simple.spc)
    else:
      dists = metrics.pairwise_distances(simple.spc,metric="cosine",n_jobs=-1)
      kmod=cluster.AgglomerativeClustering(n_clusters=npoints,affinity="precomputed",linkage="average").fit(dists)
    # from split_by_cluster()
    clusters = list(set(kmod.labels_)) # use labels_ to create a list of SpecImages
    cluster_idx=[] # will hold a list of indexes for each cluster
    for c_id in clusters: # enumerate returns a tuple (index_of_value, value)
      cluster_idx.append([i for i,j in enumerate(kmod.labels_) if j==c_id])
    for i in range(len(cluster_idx)): # cluster_idx[i] is a list (can be used to slice np.arrays, NOT lists)
      # map_idx[cluster_idx[i]] = cluster_idx[i][0] # map_idx is an np.array -> all elems map to first idx, per cluster
      valid_idx.append(location_map[cluster_idx[i][0]]) # location_map has idx from original SPC
   
    self.spc = np.array([self.spc[i] for i in valid_idx], dtype=float)
    self.idx = kmod.labels_[simple.idx] # simple.idx now has cluster ID for each new spectrum
    self.xy = simple.xy 
    return self 

  def simplify_spectral_matrix_pca(self, npoints=None, fraction_dims=None):
    """
      merge similar spectra based on their PCA dimensions (as a fraction of the total number of wave numbers). 
      Unlike simplify_spectral_matrix(), no preprocessing (like tophat correction, interpolation and smoothing) is
      applied to spectra, and no removal of noise-only spectra is  attempted.
      Updates IDX list: an index mapping pixels before and after simplification, which may be used as clustering label. 
    """
    # self.xy is not touched; self.idx will have several duplicates; self.spc will have averages.
    if npoints is None:
      npoints = int(self.spc.shape[0]/10)
    if npoints < 2:
      npoints = 2
    if npoints >= self.spc.shape[0]:
      return # more clusters than objects
    if fraction_dims is None:
      fraction_dims = 0.05
    n_dims = int(float(self.wl.shape[0]) * fraction_dims)
    if n_dims < 2:
      n_dims = 2
    transf=decomposition.PCA(n_components=n_dims).fit_transform(self.spc)
    dists = metrics.pairwise_distances(transf,metric="cosine",n_jobs=-1)
    kmod=cluster.AgglomerativeClustering(n_clusters=npoints,affinity="precomputed",linkage="average").fit(dists)
    # from split_by_cluster()
    clusters = sorted(set(kmod.labels_)) 
    cluster_idx=[] # will hold a list of indexes for each cluster
    avge_spec = []
    for c_id in clusters: # enumerate returns a tuple (index_of_value, value)
      cluster_idx.append([i for i,j in enumerate(kmod.labels_) if j==c_id])
    for i in range(len(cluster_idx)): # cluster_idx[i] is a list (can be used to slice np.arrays, NOT lists)
      avge_spec.append(np.average(self.spc[cluster_idx[i]], axis=0)) # list of arrays
   
    self.spc = np.array(avge_spec, dtype=float)
    self.idx = kmod.labels_[self.idx] # spec.idx may be already larger than original spc (previous simplifications)
    return self 

  def subtract_background_spectra(self, k_neighbours=None, weighted=None, strength=None, smooth_fraction=None, mode=None):
    """
      Finds all background spectra and creates a Voronoi diagram for them, such that each non-background spectrum uses
      only the background closer to it. This algorithm uses an efficient KDTree algorithm to find closest neighbours,
      and generally works with wavenumbers below 3000 cm-1. If the results are not satisfactory, try changing "mode"
      (between 1 and 4)
      OBS: This function currently does NOT work with arbitrary IDX vectors ("sparse spectra"). It will also in the
      future allow for smoothing the background, and dynamically calculating the background concentration (like SIS).  
    """
    ##  FIXME: weighted=True seems to be problematic
    if self.idx.shape[0] != self.spc.shape[0]:
      print ("Unfortunately the background subtraction cannot be applied to sparse spectra (that is, with arbitrary \
          pixel<->spectra mappings)")
      return
    if k_neighbours is None or k_neighbours < 1:
      k_neighbours = 1
    if weighted is None:
      weighted = False
    if smooth_fraction is None:
      smooth_fraction = 0.5
    if mode is None:
      mode = 3
    if strength is None or strength > 1 or strength < 0.0001:
      strength = 0.5
    water_idx = min(range(self.wl.shape[0]), key=lambda i: abs(self.wl[i]-3000)) # index of wl closest to 3000
    stride = int(1./smooth_fraction)
    if water_idx/stride < 10:
      stride = water_idx / 10
    if stride < 1:
      stride = 1
    # spectral matrix is smoothed, which will lead to the first clustering (using only wl<water and skipping stride elems)
    spec = np.array([moving_average(row[:water_idx], n=stride)[::stride] for row in self.spc])
    if mode != 2:
      # find which cluster, based on roughly smoothed spectra, represents background (=lower intensities excluding water region)
      kmod1 = cluster.AgglomerativeClustering(n_clusters=2, affinity='cosine', linkage='complete').fit(spec)
      back1 = np.where(kmod1.labels_ == 0)[0]; tmp = np.where(kmod1.labels_ == 1)[0]
      avge_back1 = np.average(spec[back1],1).min(); avge_tmp = np.average(spec[tmp],1).min() 
      if avge_back1 > avge_tmp: # cluster "1" has member with lower intensities than cluster "0"
        back1 = tmp; avge_back1 = avge_tmp  # we will use the average values later
    if mode > 1:
      # another clustering, based on rescaled, baseline-corrected and a different distance (Euclidean)
      spec = np.array([ndimage.morphology.white_tophat(row, None,  np.repeat([1], int(self.wl.size * 0.1) + 1)) for row in spec])
      spec = np.array([rescale_spectrum_01(row) for row in spec],dtype=float)
      kmod2 = cluster.AgglomerativeClustering(n_clusters=2, linkage='ward').fit(spec) # Ward only works with Euclidean distances
      # background RESCALED spectra are those with HIGHER intensities over region (signal-rich spectra have a few close to one)
      back2 = np.where(kmod2.labels_ == 0)[0]; tmp = np.where(kmod2.labels_ == 1)[0]
      avge_back2 = np.average(spec[back2],1).min(); avge_tmp = np.average(spec[tmp],1).min() 
      if avge_back2 < avge_tmp: # cluster "1" has higher intensities (more noisy) than cluster "0"
        back2 = tmp; avge_back2 = avge_tmp  # we will use the average values later
    if mode == 1:
      back_idx = back1
    elif mode == 2:
      back_idx = back2
    elif mode == 3:
      back_idx = np.array(sorted( set.intersection(set(back1),set(back2)) ),dtype=int) # background spectra according to both clusterings
      if back_idx.shape[0] == 0: # No elements in common between clusterings: choose the lowest values based on ORIGINAL spectra
        avge_back1 = np.average(self.spc[back1,:water_idx]); # average over flattened matrix (all values) 
        avge_back2 = np.average(self.spc[back2,:water_idx]); 
        if avge_back1 > avge_back2:
          back_idx = back2
        else:
          back_idx = back1
    elif mode == 4:
      if back1.shape[0] > back2.shape[0]:
        back_idx = back1
      else:
        back_idx = back2
    # quick search for location of nearby background spectra ( http://stackoverflow.com/a/23772423/204903 )
    tree=spatial.cKDTree(self.xy[back_idx]) # FIXME: works only when IDX is not needed
    spec = np.array(self.spc,copy=True) # new matrix, to avoid overwritting
    for i in range(self.spc.shape[0]): # includes cell *and* background
      dist, indx = tree.query(self.xy[i], k=k_neighbours)
      if (k_neighbours > 1): # dist and indx are lists
        dist = np.array(dist[np.where(indx < tree.n)[0]]) # tree.n (=self.xy[back_idx].shape[0]) means that neighbour wasn't found
        indx = np.array(indx[np.where(indx < tree.n)[0]]) # tree.n (=self.xy[back_idx].shape[0]) means that neighbour wasn't found
      else: # dist and indx are single variables
        if (indx < tree.n):
          dist=np.array([dist])
          indx=np.array([indx])
        else:
          dist=np.array([])
          indx=np.array([])
      if indx.shape[0]: # at least one neighbour was found; otherwise don't do anything
        if weighted: # indx is relative to the back_idx list, not to the whole image!
          background = np.average(self.spc[back_idx[indx]], weights = (1./(float(dist)+1e-2)), axis=0) # to avoid NaN
        else:
          background = np.average(self.spc[back_idx[indx]], axis=0)
        spec[i] = self.spc[i] - strength * background
        spec[i] -= min(spec[i]) - min(background) # maintains original scale
      # else (if indx.shape[0] is zero) then do nothing: spec[i] already has spectrum from self.spc[i]
    self.spc=spec

  def baseline_correction_polynomial(self, order=None, noise_std=None, zero=None, n_iter=None):
    """
      Finds the baseline for each spectrum by iteratively fitting a polynomial or spline and excluding values higher
      than fitted. finally subtracts this baseline signal. 
      Spline smoothing is achieved by a polynomial of order zero. (not true?)
      Alternative would be to find local minima, and/or using them in weighted splines -- see splrep() function)
    """
    if order is None:
      order=5
    elif order > 15:
      order=15
    if noise_std is None:
      noise_std = 0.1
    if zero is None:
      zero = False
    if n_iter is None:
      n_iter = len(self.wl)
    min_points = max(int(len(self.wl)/20), int(3 * order + 1)) # from spc.fit.poly.below() in hyperSpec
    for s in range(len(self.spc)):
      valid, baseline = baseline_points_by_polynomial(self.wl, self.spc[s], order=order, noise_std=noise_std,
          min_points=min_points, n_iter = n_iter)
      self.spc[s] = baseline_subtraction(self.spc[s], baseline, zero=zero)
    return self 

  def baseline_correction_rubberband (self, band_size=None, noise_std=None, zero=None):
    """
      Baseline correction based on spline smoothing through the rubberband minimum points.
    """
    if band_size is None:
      band_size = 1.
    if noise_std is None:
      noise_std = 1.
    if zero is None:
      zero = False
    for s in range(len(self.spc)):
      valid, baseline = baseline_points_by_rubberband (self.wl, self.spc[s], band_size=band_size, noise_std=noise_std)
      # discard rubberband baseline above, which is just piecewise minima.
      baseline = interpolate.splev(self.wl, interpolate.splrep(self.wl[valid],self.spc[s,valid]), der=0) 
      self.spc[s] = baseline_subtraction(self.spc[s], baseline, zero=zero)
    return self 

  def baseline_correction_als(self, smooth=None, asym=None, n_iter=None, zero=None):
    """
    http://stackoverflow.com/questions/29156532/python-baseline-correction-library
        "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005. 
        "smooth" is log(lambda) and "asym" is p in the paper (that is, lambda between 7 and 20 are equiv to 10^3 ~ 10^9)
    """
    if (smooth is None) or (smooth < 1):
      smooth = 10
    if (asym is None) or (asym > 0.999) or (asym < 1e-5):
      asym = 0.01
    if n_iter is None:
      n_iter = 20
    if zero is None:
      zero = False
    L = len(self.wl)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    for s in range(len(self.spc)):
      valid, baseline = baseline_points_by_als (self.wl, self.spc[s], smooth=smooth, asym=asym, n_iter=n_iter)
      self.spc[s] = baseline_subtraction(self.spc[s], baseline, zero=zero)
    return self 

  def baseline_correction_tophat(self, f = None):
    if (f == None) or (f < 0.000001) or (f > 1.):
      footp = np.repeat([1], int(self.wl.size * 0.4) + 1)
    else:
      footp = np.repeat([1], int(self.wl.size * f) + 1)
    min_spec = self.spc.min(1) # 1 = min() per row; 0 = min() per col || tophat forces min() to zero
    self.spc = np.array([ndimage.morphology.white_tophat(spec, None, footp) + z for spec,z in zip(self.spc,min_spec)],dtype=float)
    return self
  
  def background_correction_remove_percentile(self, percentile=None, flatten=None):
    """
    Calculate the x% lower intensity value over all samples per wavelength, and then for each spectrum subtract these
    values -- that is, create a spectrum of "minima", representing the background, that should be discounted
    Warning: here "percentile" is a single parameter 
    """
    if percentile is None:
      percentile = 1
    if flatten is None:
      flatten = False
    background = np.percentile(self.spc,q=percentile,axis=0) # axis=0 -> over wavelengths
    self.spc=np.array([spec - background for spec in self.spc],dtype=float)
    if flatten:
      self.spc[ self.spc < 0.] = 0.
    return self

  def background_correction_remove_minimum(self):
    background = np.min(self.spc,axis=0) # axis=0 -> over wavelengths
    self.spc=np.array([spec - background for spec in self.spc],dtype=float)
    return self

  def background_correction_emsc(self, order=None, fit=None):
      """Extended Multiplicative Scatter Correction (Ref H. Martens)
      order -     order of polynomial
      fit -       if None then use average spectrum, otherwise provide a spectrum
                  as a column vector to which all others fitted
      --          imported from PyChem http://pychem.sourceforge.net/ (Roger Jarvis, GPL)
      """
      if fit is None: #  fitting spectrum
        mx = np.mean(self.spc,axis=0)[:,np.newaxis]
      else:
        mx = fit
      if order is None:
        order = 2
      corr = np.zeros(self.spc.shape)
      for i in range(len(self.spc)): # do fitting
        b,f,r = pychem_mlr (mx, self.spc[i,:][:,np.newaxis], order)
        corr[i,:] = np.reshape((r/b[0,0]) + mx, (corr.shape[1],))

      self.spc = corr
      return self

  def background_correction_pca(self, n_components=None, approximate=None):
    """ 
    Based on PCA Noise Reduction technique, that aims at re-weighting wavelengths based on how much they contribute to
    variance (PCA-based dimensionality reduction). This function also finds non-informative pixels, besides wavelengths.
    n_components[] determine how many PCs are used over wavelengths [0] and over pixels [1], with values too large or
    zero indicating to skip this step. 
    Must use carefully since assumes conserved values are not informative (for this image), which might remove signal
    relevant when comparing images.
    """
    if n_components is None:
      n_components = [5,5]
    if n_components[0] < self.spc.shape[1] and n_components[0] > 0:
      if approximate is None or approximate is False:
        pca=decomposition.PCA(n_components=n_components[0])
      else:
        pca=decomposition.RandomizedPCA(n_components=n_components[0])
      Y = pca.fit_transform(self.spc) # dimensions are wavelengths and samples are pixels 
      self.spc = pca.inverse_transform(Y) # reconstruct original data using only "n_components" projections 
    # do same thing across pixels instead of wavelengths 
    if n_components[1] < self.spc.shape[0] and n_components[1] > 0:
      if approximate is None or approximate is False:
        pca=decomposition.PCA(n_components=n_components[1])
      else:
        pca=decomposition.RandomizedPCA(n_components=n_components[1])
      Y = pca.fit_transform(self.spc.T) # dimensions are now the pixels and "samples" are the wavelengths
      self.spc = pca.inverse_transform(Y).T # transpose again, to have correct matrix dimension
    return self

  def find_endmembers(self, n_members=None, method=None):
    """
      Extraction of endmembers (a.k.a. pure spectra) using pysptools library (http://pysptools.sourceforge.net/eea_fn.html)
      Possible methods are "nfindr", "ppi" (pixel purity index), "fippi"(Fast Iterative Pixel Purity Index), and
      "atgp"(Automatic Target Generation Process). 
      This function will return a numpy array with n_members endmembers and a vector with indexes of such endmembers,
      when available (that is, for all except PPI.) 
      OBS: FIPPI method does not respect n_members, can give larger value!
    """
    try:
      from pysptools.eea import eea, nfindr
    except ImportError:
      sys.stderr.write("You don't seem to have http://pysptools.sourceforge.net/ properly installed. Sorry.\n")
      return np.array([]), np.array([])
    if n_members is None:
      n_members = 4
    if method is "atgp":
      em, idx = eea.ATGP (self.spc, n_members)
    elif method is "fippi":
      em, idx = eea.FIPPI (self.spc, n_members)
    elif method is "ppi": 
      em = eea.PPI (self.spc, n_members, 1000) # suggested numSkewers is 10k
      idx = []
    else: # "nfindr" or None
      em, _, idx, _ = nfindr.NFINDR (self.spc, n_members)
    return em, idx

  def find_abundance_maps(self, endmembers=None, method=None, normalize=None, closure=None, reorder=None):
    """
      Abundance maps (contribution of each endmember) for each spectrum using pysptools library
      (http://pysptools.sourceforge.net/abundance_maps_fn.html)
      Possible methods are "fcls" (fully constrained least squares), "nnls" (non-negative constrained least squares),
      and "ucls" (unconstrained least squares) abundance estimation. FCLS is least squares with the abundance
      sum-to-one constraint (a.k.a. closure) and the abundance nonnegative constraint, while NNLS has only the abundance 
      nonnegative constraint. FCLS may return an error, therefore there is an option to rescale the abundances to the 
      interval  zero-one.
      If no method is chosen, then the standard non-negative least squares from the SciPy library is used -- it should
      be equivalent to "nnls" and is the one used by MCR.
      Returns a numpy array with the abundances of each endmember for all spectra, one spectrum per row. Also returns
      the received "endmembers" array. If this array -- from a previous call to specimage.find_endmembers() -- is absent, 
      this function will create it by calling specimage.find_endmembers() 
    """
    try:
      import pysptools.abundance_maps.amaps as amaps
    except ImportError:
      sys.stderr.write("I can't find http://pysptools.sourceforge.net, so I'll use scypy.optimize.nnls()\n")
      abundance = np.array([optimize.nnls(endmembers.T, self.spc[i,:])[0] for i in range(self.spc.shape[0])])
      if closure:
        abundance = np.array([(i/np.sum(i)) for i in abundance])
      return abundance, endmembers, resid
    if endmembers is None:
      endmembers = self.find_endmembers()
    if normalize:
      endmembers = np.array([spec/np.sum(spec) for spec in endmembers],dtype=float)
    if method is "fcls":
      abundance = amaps.FCLS(self.spc, endmembers)
    elif method is "ucls": # UCLS can return negative abundances
      abundance = amaps.UCLS(self.spc, endmembers)
    elif method is "nnls":
      abundance = amaps.NNLS(self.spc, endmembers)
    else: # None
      abundance = np.array([optimize.nnls(endmembers.T, self.spc[i,:])[0] for i in range(self.spc.shape[0])])

    if closure:
      abundance = np.array([(i/np.sum(i)) for i in abundance])
    
    if reorder:
      ordered_idx = np.argsort(abundance.sum(0))[::-1] # indexes of spectra from largest to lowest
      endmembers = endmembers[ordered_idx] # reorder rows s.t. first endmembers have highest peaks 
      abundance = abundance[:,ordered_idx] # reorder columns of abundances to follow order of endmembers

    resid = np.array([self.spc[i,:] - np.dot(abundance[i,:], endmembers) for i in range(self.spc.shape[0])])
    return abundance, endmembers, resid

  def find_endmember_abundance_MCR (self, n_members=None, abundances=None, endmembers=None, threshold=None, 
      max_iter=None, randomize=None, debug=None, reorder=None, closure=None, normalize=None):
    """
     Estimation of endmembers and abundance maps by multivariate curve resolution alternating least squares (MCR-ALS).
     Code adapted from ALS for R (https://cran.r-project.org/web/packages/ALS/index.html)
    "n_members" or "endmembers" should be present. If endmembers is absent, it will estimate the initial
     composition through FIPPI (or randomly from the data). 
     Returns the abundances, endmembers and the residuals (for whole spectral data)
     1) force all initial endmembers and abundances to be larger than zero
     2) the endmembers (and abundances) can be reordered at the end 
     3) if max_iter is chosen, then this function will iterate accordingly, even after reaching convergence.
     * Idea: force some endmembers to be fixed at initial state (like known pure spectra)
     * WARNING: Normalization (endmembers with sum = 1) and closure (abundances with sum=1) must be used carefully when
     together, since they provide very poor fit to unscaled spectra -- and not unlikely will result in non-valid
     values and errors. 
    """
    if threshold is None:
      threshold = 0.001
    if max_iter is None:
      max_iter  = 100
      force_maxiter = False 
    else:
      force_maxiter = True 
    if randomize is None:
      randomize = False
    if closure is None:
      closure = False
    if normalize is None:
      normalize = False
    this_iter = max_iter
    if endmembers is not None and n_members is not None and n_members > endmembers.shape[0]:
    # partial MCR where some endmembers are fixed and others must be estimated
      n_fixed = endmembers.shape[0]
      new_em = self.find_endmembers(n_members=n_members - n_fixed, method="nfindr")[0] # does it work for one endmember?
      endmembers = np.append(endmembers, new_em, axis=0)
      if abundances is not None:
        if abundances.shape[1] != n_fixed:
          sys.stderr.write("Sizes of abundances and endmembers disagree in partial MCR. Calculating from scratch\n")
          abundances = np.array([optimize.nnls(endmembers.T, self.spc[i,:])[0] for i in range(self.spc.shape[0])])
        else:
          # fixed_spc is the spectral part explained by fixed endmembers
          fixed_spc = np.array([self.spc[i,:] - np.dot(abundances[i,:], endmembers[:n_fixed]) for i in range(self.spc.shape[0])])
          new_abund = np.array([optimize.nnls(new_em.T, fixed_spc[i,:])[0] for i in range(fixed_spc.shape[0])])
          abundances = np.append(abundances, new_abund, axis=1) 
    else: # i.e. no partial MCR
      n_fixed = False
    if endmembers is None:
      if n_members is None:
        n_members = 4
      if randomize:
        endmembers = np.array( self.spc[np.random.choice(self.spc.shape[0], size=n_members, replace=False)] )
      else:
        endmembers = self.find_endmembers(n_members=n_members, method="nfindr")[0]
    if abundances is None:
      if randomize: # to replicate parallel version
        #abundances = np.ones((self.spc.shape[0],endmembers.shape[0])) 
        abundances = np.random.random_sample((self.spc.shape[0],endmembers.shape[0])) 
      else:
        abundances = np.array([optimize.nnls(endmembers.T, self.spc[i,:])[0] for i in range(self.spc.shape[0])])
    
    if endmembers.min() < 0.:
      endmembers -= endmembers.min() # cannot have negative numbers
    if abundances.min() < 0.:
      abundances -= abundances.min()  # abundances in particular cannot be all zero either
    if normalize is True:
      endmembers = np.array([spec/np.sum(spec) for spec in endmembers],dtype=float)
    if closure is True:
      abundances = np.array([(i/np.sum(i)) for i in abundances])

    # residuals for the initial configuration 
    resid = np.array([self.spc[i,:] - np.dot(abundances[i,:], endmembers) for i in range(self.spc.shape[0])])
    initial_rss = old_rss = np.sum(np.power(resid,2)) # residual sum of squares
    for iteration in range(max_iter):
      if n_fixed:
        fixed_spc = np.array([self.spc[i,:] - np.dot(abundances[i,:n_fixed], endmembers[:n_fixed]) for i in range(self.spc.shape[0])])
        new_em = np.array([optimize.nnls(abundances[:,n_fixed:], fixed_spc[:,i])[0] for i in range(fixed_spc.shape[1])]).T
        endmembers = np.append(endmembers[:n_fixed], new_em, axis=0)
      else: # no fixed endmembers
        endmembers = np.array([optimize.nnls(abundances,   self.spc[:,i])[0] for i in range(self.spc.shape[1])]).T
      if normalize is True:
        endmembers = np.array([spec/np.sum(spec) for spec in endmembers],dtype=float)
      abundances = np.array([optimize.nnls(endmembers.T, self.spc[i,:])[0] for i in range(self.spc.shape[0])])
      if closure is True:
        abundances = np.array([(i/np.sum(i)) for i in abundances])

      resid = np.array([self.spc[i,:] - np.dot(abundances[i,:], endmembers) for i in range(self.spc.shape[0])])
      rss = np.sum(np.power(resid,2)) # residual sum of squares
      if not force_maxiter and (old_rss - rss)/old_rss < threshold:
        this_iter = iteration
        break # finished
      old_rss = rss
    sys.stdout.write("finished in " + str(iteration) + " iterations\n");
    """
        OBS from importing ALS from R:
    * PsiList[nsamples, wl]  -> spc || CList[nsamples, n_end] -> abundance ||  S[wl, n_end] -> endmembers 
    * endmembers = t(S) -- that is, in R (nrows,ncols)= (WL,n_endmembers) while here is opposite
    * we don't use weights (WList[] in R), but they would be like:
       resid = np.array([spc[i,:] - np.dot(abundances[i,:], endmembers * weights[i,:]) for i in range(spc.shape[0])])
    * original ALS code has one_more_iter that depends on closureC and normS that is, must optimize endmembers and abundances once more)
    """
    if reorder:
      ordered_idx = np.argsort(np.max(endmembers,1))[::-1] # indexes of spectra from largest to lowest
      endmembers = endmembers[ordered_idx] # reorder rows s.t. first endmembers have highest peaks 
      abundances = abundances[:,ordered_idx] # reorder columns of abundances to follow order of endmembers

    if debug:
      return abundances, endmembers, resid, initial_rss, rss, this_iter
    else:
      return abundances, endmembers, resid

  def find_endmember_abundance_MCR_parallel (self, n_members=None, abundances=None, endmembers=None, threshold=None, 
      max_iter=None, n_chunks=None, force_maxiter=None, debug=None, reorder=None):
    """
     Estimation of endmembers and abundance maps by multivariate curve resolution alternating least squares (MCR-ALS).
     Code adapted from ALS for R (https://cran.r-project.org/web/packages/ALS/index.html)
    "n_members" or "endmembers" should be present. If endmembers is absent, it will sample random spectra from image. 
     Returns the abundances, endmembers and the residuals (for whole spectral data)
     Multithreaded version.
    """
    import multiprocessing
    if force_maxiter is None:
      force_maxiter = False
    if threshold is None:
      threshold = 0.001
    if max_iter is None:
      max_iter  = 100
    this_iter = max_iter
    if n_members is None:
      n_members = 4
    if endmembers is None:
      endmembers = np.array( self.spc[np.random.choice(self.spc.shape[0], size=n_members, replace=False)] )
    if abundances is None:
      abundances = np.random.random_sample((self.spc.shape[0],endmembers.shape[0])) 
    if n_chunks is None:
      n_chunks = multiprocessing.cpu_count()
    
    endmembers -= (endmembers.min() - 0.01)  # cannot have negative numbers
    abundances -= (abundances.min() - 0.01)  # abundances in particular cannot be all zero either
    ival = np.linspace(0,self.spc.shape[1],n_chunks+1); 
    i_wl = [[int(ival[i]),int(ival[i+1])] for i in range(len(ival) - 1)]
    ival = np.linspace(0,self.spc.shape[0],n_chunks+1); 
    i_px = [[int(ival[i]),int(ival[i+1])] for i in range(len(ival) - 1)]

    pool = multiprocessing.Pool()

    # residuals for the initial configuration 
    resid = np.array([self.spc[i,:] - np.dot(abundances[i,:], endmembers) for i in range(self.spc.shape[0])])
    initial_rss = old_rss = np.sum(np.power(resid,2)) # residual sum of squares
    for iteration in range(max_iter):
      sys.stdout.write("iteration " + str(iteration) + ": endmembers "); sys.stdout.flush()
      params = [[abundances, np.array([self.spc[:,i] for i in range(i_wl[j][0],i_wl[j][1])])] for j in range(len(i_wl))]
      endmembers_list = pool.map(optim_nnls_parallel, params)
      endmembers = np.concatenate([j.T for j in endmembers_list], 1)
      sys.stdout.write(" and abundances\n");
      params = [[endmembers.T, np.array([self.spc[i,:] for i in range(i_px[j][0],i_px[j][1])])] for j in range(len(i_px))]
      abundances_list = pool.map(optim_nnls_parallel, params)
      abundances = np.concatenate(abundances_list, 0)
      resid = np.array([self.spc[i,:] - np.dot(abundances[i,:], endmembers) for i in range(self.spc.shape[0])])
      rss = np.sum(np.power(resid,2)) # residual sum of squares
      if not force_maxiter and (old_rss - rss)/old_rss < threshold:
        this_iter = iteration
        break # finished
      old_rss = rss

    if reorder:
      ordered_idx = np.argsort(np.sum(endmembers,1)) # indexes of spectra in decreasing order
      endmembers = endmembers[ordered_idx] # reorder rows s.t. first endmembers have lower integral
      abundances = abundances[:,ordered_idx] # reorder columns of abundances to follow order of endmembers
    pool.close()
    if debug:
      return abundances, endmembers, resid, initial_rss, rss, this_iter
    else:
      return abundances, endmembers, resid

  
  def histogram_wl_imshow(self, cut_percent=None, index=None, nbins=None, rescale_column=None):
    """
    Histogram of intensities for each wavelength, where range of plotted intensities exclude outliers (defined by
    "cut_percent"). If index[] is present, only these elements from spectral matrix are included in histogram (but not
    in Y axis range, which includes all pixels s.t. several plots can use same axes.)
      -- for use like imshow(hst, aspect='auto', cmap=plt.get_cmap("Blues"), extent=extent)

      OBS: IT ASSUMES THERE ARE NO GAPS in WAVELENGTHS (that is, no silent region removal etc.)
    """
    if nbins == None:
      nbins=49
    if cut_percent is None:
      cut_percent = 5
    extent=[min(self.wl),max(self.wl), 
        np.min(np.percentile(self.spc,q=cut_percent,axis=0)), np.max(np.percentile(self.spc,q=100-cut_percent,axis=0))]
    binlist=np.linspace(extent[2],extent[3],nbins+1)
    if index is not None:
      histog=np.array([np.histogram(col, bins=binlist)[0] for col in self.spc[index].transpose()],dtype=float)
    else:
      histog=np.array([np.histogram(col, bins=binlist)[0] for col in self.spc.transpose()],dtype=float)
    # now each row of hist has intensity counts per wavelength (we need the transpose -> changes rows and columns)
    # hist[] has "wl" rows and "nbin" cols
    valid_bins = np.array([i for i in range(nbins) if np.sum(histog[:,i]) >= 1]) # find all-zero bins 
    if valid_bins.shape[0] < nbins:
      full_bins = np.linspace(extent[2], extent[3], nbins) # notice that is one less than for the histogram
      histog = np.array([np.interp(full_bins, full_bins[valid_bins], row[valid_bins]) for row in histog],dtype=float)
    # makes wavenumbers evenly spaced (assumes NO missing data like e.g. removed silent region)
    full_wl = np.linspace(extent[0], extent[1], 2 * self.wl.shape[0])
    histog = np.array([np.interp(full_wl, self.wl, row) for row in histog.transpose()],dtype=float).transpose()
    
    if rescale_column:
      histog=np.fliplr([row/max(row) for row in histog]) # now each wavelength will have hist=1 for max value, w/ intensity from 0 to one
    else:
      max_all = np.max(histog)
      histog=np.fliplr([row/max_all for row in histog]) # now each wavelength will have hist=1 for max value, w/ intensity from 0 to one

    return histog.transpose(), extent
    

  def histogram_wl_imshow_old(self, cut_percent=None, index=None, nbins=None):
    """
    Histogram of intensities for each wavelength, where range of plotted intensities exclude outliers (defined by
    "cut_percent"). If index[] is present, only these elements from spectral matrix are included in histogram (but not
    in Y axis range, which includes all pixels s.t. several plots can use same axes.)
      -- for use like imshow(hst, aspect='auto', cmap=plt.get_cmap("Blues"), extent=extent)
    """
    if nbins == None:
      nbins=49
    if cut_percent is None:
      cut_percent = 5
    extent=[min(self.wl),max(self.wl), 
        np.min(np.percentile(self.spc,q=cut_percent,axis=0)), np.max(np.percentile(self.spc,q=100-cut_percent,axis=0))]
    binlist=np.linspace(extent[2],extent[3],nbins+1)
    if index is not None:
      hist=np.array([np.histogram(col, bins=binlist)[0] for col in self.spc[index].transpose()],dtype=float)
    else:
      hist=np.array([np.histogram(col, bins=binlist)[0] for col in self.spc.transpose()],dtype=float)
    # now each row of hist has intensity counts per wavelength (we need the transpose -> changes rows and columns)
    hist=np.fliplr([row/max(row) for row in hist]) # now each wavelength will have hist=1 for max value, w/ intensity from 0 to one
    # if we ever need the bins, it's np.linspace(0,1,nbins) --> notice that is one less than for the histogram
    # incomplete hist is hist.transpose(), but maybe some columns are absent
    dif = min(np.diff(self.wl))
    full_wl = np.arange(extent[0], extent[1], step=dif)
    full_h = hist[0]
    count = 1
    for nwl in full_wl:
      if np.absolute(nwl - self.wl[count]) < dif:
        full_h = np.vstack([full_h, hist[count]]);
        count += 1
      else:
        full_h = np.vstack([full_h, np.zeros(nbins)]);
    return full_h.transpose(), extent
    
  def histogram_wl(self, is_scaled=None, nbins=None):
    """
    Histogram of intensities for each wavelength, scaled s.t. mode=1 
      -- for use like imshow(hst, aspect='auto', cmap=plt.get_cmap("Blues"), extent=[min(a.wl),max(a.wl),0,1])
    """
    if nbins == None:
      nbins=49
    if is_scaled==None:
      is_scaled=False
    if is_scaled is True:
      spectra=self.spc # view (shallow copy)
    else:
      spectra=np.array([rescale_spectrum_01(spec) for spec in self.spc],dtype=float)
    binlist=np.linspace(0,1,nbins+1)
    hist=np.array([np.histogram(col, bins=binlist)[0] for col in spectra.transpose()],dtype=float)
    # now each row of hist has intensity counts per wavelength (we need the transpose -> changes rows and columns)
    hist=np.fliplr([row/max(row) for row in hist]) # now each wavelength will have hist=1 for max value, w/ intensity from 0 to one
    # if we ever need the bins, it's np.linspace(0,1,nbins) --> notice that is one less than for the histogram
    # incomplete hist is hist.transpose(), but maybe some columns are absent
    dif = min(np.diff(self.wl))
    full_wl = np.arange(min(self.wl), max(self.wl), step=dif)
    full_h = hist[0]
    count = 1
    for nwl in full_wl:
      if np.absolute(nwl - self.wl[count]) < dif:
        full_h = np.vstack([full_h, hist[count]]);
        count += 1
      else:
        full_h = np.vstack([full_h, np.zeros(nbins)]);
    return full_h.transpose()
    
  def idx_ordered_by_intensity(self, logscale=None):
    if logscale is None or logscale is True:
      intensity=[sum(np.log(spec - min(spec) + 1e-3)) for spec in self.spc] # one value per spectrum (and avoiding log(zero))
    else:
      intensity=[sum(spec) for spec in self.spc] # one value per spectrum
    return np.argsort(intensity)[::-1] # reverse order

  def percentile(self, percentiles=None, do_scaling=None):
    """
    Returns (nxW) array, with chosen n quantiles of intensities per wavelength 
      -- for use like plot(self.wl, med[0], "b-") 
    Warning: here "percentile" is a list, even if with one single entry, as in [50]
    CORRECTION: it seems that np.percentile's behaviour changed; now single value is not list!
    """
    if percentiles is None:
      percentiles = 50
    if do_scaling == None:
      spectra=self.spc # view (shallow copy)
    else:
      spectra=np.array([rescale_spectrum_01(spec) for spec in self.spc],dtype=float)
    return np.percentile(spectra,q=percentiles,axis=0) # axis=0 -> over wavelengths

  def quantile_lineplot(self, index=None, exclude_minmax=None, colrange=None, nbins=None, cmap=None, lwd=None):
    """
    Returns LineCollection  with quantiles of intensities per wavelength, with transparency
    Also returns xlims and ylims (min and max intensities), like the "extents" of imshow()
    If index[] list is present, only these elements from compacted spc will be used
      -- for use like add_collection(qline) 
        from   http://exnumerus.blogspot.com.es/2011/02/how-to-quickly-plot-multiple-line.html
    """
    if exclude_minmax is None:
      exclude_minmax = True
    if nbins is None:
      nbins=5
    if lwd is None:
      lwd=1
    if cmap is None:
      cmap = "Reds"
    if colrange is None:
      colrange=[0.2,1]
    if exclude_minmax:
      quants=np.linspace(0,100,nbins+2)[1:-1] 
    else: 
      quants=np.linspace(0,100,nbins) 
    if index is not None:
      spectra = np.array(self.spc[index]) # view (shallow copy)
    else:
      spectra = np.array(self.spc) # view (shallow copy)
    # in line below, "q" must be a python array, not a numpy.array()
    ysegs=np.percentile(spectra,q=[i for i in quants],axis=0) # axis=0 -> over wavelengths
    quants=quants*quants[::-1] # triangle with peak around 0.5
    quants=(quants - min(quants))/(max(quants)-min(quants)) * (colrange[1] - colrange[0]) + colrange[0]# rescaling
    # change cmap so that colors close to white become transparent
    cmaphere=plt.get_cmap(cmap)
    cmaphere._init()
    alphas=np.linspace(0.25, 1, cmaphere.N+3)
    cmaphere._lut[:,-1]=alphas # equiv to to_rgba(); lut[] has four values RGBA
    # each percentile is a line
    lc=LineCollection([list(zip(self.wl,y)) for y in ysegs], cmap=cmaphere, linewidths=lwd, norm=plt.Normalize(0,1))
    lc.set_array(quants) # colors follow quants values according to cmap
    return lc, [min(self.wl), max(self.wl),np.min(ysegs),np.max(ysegs)] 
    
  def split_by_cluster(self, cluster_labels):
    clusters=list(set(cluster_labels))
    idx=[] # will hold indexes for each cluster
    for c_id in clusters:
      idx.append([i for i,j in enumerate(cluster_labels) if j==c_id])
    cls = [SpecImage() for i in range(len(idx))] # list of SpecImage objects
    for i in range(len(idx)):
      cls[i].wl  = np.array(self.wl)
      cls[i].xy  = np.array(self.xy[idx[i]])
      cls[i].idx = np.array(self.idx[idx[i]]) # correspond to indexes in original (self) specimage object
      cls[i].spc = np.array(self.spc[idx[i],:])
    return cls

  def append_specimage(self, atmp, location=None, rescale=None):
    """
      new images canbe added to the right or to the top (due to x and y convention of quadrants) of existing image.
      This function should be used only for "simple" concatenation of images (all horizontal or all vertical). Complex
      configurations (like matrices 2x2) can be achieved by 1st creating simple "superimage" slices and then concatenating
      them.
    """
    if rescale is True:
      rescale = [1,1]
    if self.spc.size:
      if self.spc.shape[0] != max(self.idx) + 1: #DEBUG
        print ("ERROR: IDX does not correspond to SPC location in " + self.description)
      # just concatenate spectra, possibly after interpolating s.t. WL are the same (endpoints assumed to be similar)
      diff_wl=np.sum([(i-j)**2 for i,j in zip(self.wl,atmp.wl)]) # eq to ((x-y)**2).sum() but works for unequal sizes
      if self.wl.shape[0] != atmp.wl.shape[0] or diff_wl > 0.01: # must interpolate to same WL as self 
        aspc = np.array([np.interp(self.wl, atmp.wl, row) for row in atmp.spc],dtype=float)
      else:
        aspc = atmp.spc
      # IDX of new spectra must be shifted to reflect concatenation
      aidx = atmp.idx + self.spc.shape[0]

      # find where new XY values should be 
      axy = np.array(atmp.xy)
      min_x2 = axy[:,0].min(); max_x2 = axy[:,0].max() # corners of second image
      min_y2 = axy[:,1].min(); max_y2 = axy[:,1].max()
      try:  ## if atmp.xy_null is not None:
        min_x2 = np.min([min_x2, atmp.xy_null[:,0].min()]); max_x2 = np.max([max_x2, atmp.xy_null[:,0].max()]); 
        min_y2 = np.min([min_y2, atmp.xy_null[:,1].min()]); max_y2 = np.max([max_y2, atmp.xy_null[:,1].max()]); 
      except AttributeError:
        pass
      except TypeError: # complaint: 'NoneType' object has no attribute '__getitem__'
        pass
      except IndexError: # when trying to access index of xy_null[]
        pass
      min_x = self.xy[:,0].min(); min_y = self.xy[:,1].min(); ## corners of first image 
      try:  ## if self.xy_null is not None:
        min_x = np.min([min_x, self.xy_null[:,0].min()]); min_y = np.min([min_y, self.xy_null[:,1].min()]);
      except AttributeError:
        pass
      except TypeError:
        pass
      except IndexError: # when trying to access index of xy_null[]
        pass
      if min_x or min_y: # image must be moved to origin  
        self.xy[:,0] -= min_x # both dimensions should start at zero 
        self.xy[:,1] -= min_y 
      max_x = self.xy[:,0].max();max_y = self.xy[:,1].max() ## corners of first image
      try: ## if self.xy_null is not None:
        max_x = np.max([max_x, self.xy_null[:,0].max()]); max_y = np.max([max_y, self.xy_null[:,1].max()]); 
      except AttributeError:
        pass
      except TypeError:
        pass
      except IndexError:
        pass
      
      # to avoid overlap, corner of second image must be max of first image plus epslon
      max_x += np.diff(sorted(set(self.xy[:,0]))).min()
      max_y += np.diff(sorted(set(self.xy[:,1]))).min()

      axy[:,0] -= min_x2 # both dimensions should start at zero 
      axy[:,1] -= min_y2
      if rescale: # change coordinates to be between zero and rescale[]
        axy[:,0] = rescale[0] * axy[:,0]/max_x2
        axy[:,1] = rescale[1] * axy[:,1]/max_y2 
      if location is None or location == 0: # concatenate horizontally
        axy[:,0] += max_x # stack X values of different images, leaving Y at their zero-origin values
      else: # concatenate vertically
        axy[:,1] += max_y # stack Y values, while X of both images start at zero

      try: ## if atmp.xy_null is not None:
        axynul = np.array(atmp.xy_null)
        axynul[:,0] -= min_x2 # both dimensions should start at zero 
        axynul[:,1] -= min_y2
        if rescale: # change coordinates to be between zero and rescale[]
          axynul[:,0] = rescale[0] * axynul[:,0]/max_x2
          axynul[:,1] = rescale[1] * axynul[:,1]/max_y2 
        if location is None or location == 0: # concatenate horizontally
          axynul[:,0] += max_x # stack X values of different images, leaving Y at their zero-origin values
        else: # concatenate vertically
          axynul[:,1] += max_y # stack Y values, while X of both images start at zero
      except AttributeError: 
        axynul = None
      except TypeError:
        pass
      except IndexError: # if xy_null is None then axynul[:,0] won't work: 0-d array
        pass

      self.xy = np.concatenate((self.xy, axy))
      self.spc = np.concatenate((self.spc,aspc))
      self.idx = np.concatenate((self.idx,aidx))
      try: ## if atmp.xy_null is not None: if self.xy_null is not None:
        self.xy_null = np.concatenate((self.xy_null, axynul))
      except ValueError: # not AttributeError, but "zero-dimensional arrays cannot be concatenated"
        self.xy_null = axynul

    else: ## if self.spc.size
      self.spc = np.array(atmp.spc, copy=True)
      self.idx = np.array(atmp.idx, copy=True) 
      self.xy = np.array(atmp.xy, copy=True)
      self.wl = np.array(atmp.wl, copy=True) 

      min_x = self.xy[:,0].min(); max_x = self.xy[:,0].max()
      min_y = self.xy[:,1].min(); max_y = self.xy[:,1].max()
      try: ## if atmp.xy_null is not None:
        min_x = np.min([min_x, atmp.xy_null[:,0].min()]); max_x = np.max([max_x, atmp.xy_null[:,0].max()]); 
        min_y = np.min([min_y, atmp.xy_null[:,1].min()]); max_y = np.max([max_y, atmp.xy_null[:,1].max()]); 
        self.xy_null = np.array(atmp.xy_null, copy=True) 
        self.xy_null[:,0] -= min_x # both dimensions should start at zero 
        self.xy_null[:,1] -= min_y 
        if rescale: # change coordinates to be between zero and rescale[]
          self.xy_null[:,0] = rescale[0] * self.xy_null[:,0]/max_x
          self.xy_null[:,1] = rescale[1] * self.xy_null[:,1]/max_y
      except AttributeError:
        pass
      except TypeError:
        pass
      except IndexError:
        pass
      self.xy[:,0] -= min_x # both dimensions should start at zero 
      self.xy[:,1] -= min_y 
      if rescale: # change coordinates to be between zero and rescale[]
        self.xy[:,0] = rescale[0] * self.xy[:,0]/max_x
        self.xy[:,1] = rescale[1] * self.xy[:,1]/max_y
    if self.spc.shape[0] != max(self.idx) + 1: #DEBUG
      print ("ERROR: IDX does not correspond to SPC location in " + self.description)

  def peak_areas_only(self, f = None, neighbours=None, use_min=None):
    if (f is None):
      f = 0.01
    if (neighbours is None):
      neighbours = 0
    if (use_min is None):
      use_min = False
    w = int(self.wl.size * f) + 1
    # axis=1 -> use neighbors from same row; [1] are the wl indexes (and [0] are pixels) 
    peaks = signal.argrelmax(self.spc,order=w, axis=1)[1] 
    if use_min:
      peaks = np.append(peaks, signal.argrelmin(self.spc,order=w, axis=1)[1])
    peaks = np.unique(peaks)
    wlist = np.arange(-neighbours, neighbours+1) 
    valid=[]
    for w in wlist: # use areas around it
      valid = np.append(valid, peaks + w)
    valid = sorted(np.unique(valid))
    self.spc = self.spc[:,valid]
    self.wl = self.wl[valid]
    return self

  def diff(self):
    self.spc = np.array([np.diff(spec) for spec in self.spc], dtype=float)
    self.wl = (self.wl[1:] + self.wl[:-1])/2
    return self

def read_pickle(filename, update=None):
  fl=gzip.open(filename, "r")
  a=pickle.load(fl, encoding='latin1')
  fl.close()
  if update is None: # just assume the pickled object is up-to-date (i.e. not from an old SpecImage version)
    return a
  if isinstance(a,list): # update a list of SpecImage objects
    this = [SpecImage(copyfrom=spc) for spc in a]
  else: # pickled object contains just one SpecImage
    this = SpecImage(copyfrom=a)
  return this

def write_pickle(thisvar, filename):
  fl=gzip.open(filename, "w")
  pickle.dump(thisvar,fl,2)
  fl.close()

def rescale_spectrum_01(spec):
    minx = min(spec)
    ranx = max(spec) - minx
    return (spec - minx)/ranx

def factorise (n, lower=None): 
  if (lower is None): #return only factors larger than sqrt()
    return set(reduce(list.__add__, ([n//i] for i in range(1, int(np.sqrt(n)) + 1) if n % i == 0)))
  else:  
    return set(reduce(list.__add__, ([i] for i in range(1, int(np.sqrt(n)) + 1) if n % i == 0)))

def moving_average (series, n = None, power=None, weights=None):
  if n is None or n < 2:
    n = 2
  if power is None:
    power = 2.
  n = int(n)
  if weights is None:
    weights = np.power( np.convolve (np.repeat(1./float(n), n), np.repeat(1./float(n), n)),power) # exponentiated triangular weights
    weights /= sum(weights) # must always sum up to one
  series = np.append(np.repeat(series[0], n-1), series)   # moving average will reduce series length by (n-1) ,
  series = np.append(series, np.repeat(series[-1], n-1))  # and convolve will generate weight array of size 2n-1
  return np.correlate(series, weights) 

def create_grid_imshow(xy, z, xy_null=None, null_zvalue=None, resolution=None, interp=None):
  """
    Receives XY matrix as from SpecImage class (one pair of values per row) together with Z values (to be colored by
    heatmap, e.g.). Returns griddata with x,y,z values with interpolation for missing xy values. Also returns "extent"
    list that defines borders for plt.imshow()
    The parameter "interp" describes the method for interpolation: "nearest", "linear", or "cubic".
    Remember that if Z was calculated over spectra it will have (many) fewer elements than XY. To account for that you
    may need to use " z[self.idx] " (where "self" is your specimage object.) Furthermore if xy_null[] list is present,
    it is assumed to contain coordinates removed from SpecImage (that is, not associated to any spectra), and will be
    represented by value "null_zvalue"
  """
  if resolution is None:
    resolution = 200
  xy=np.array(xy, dtype=float)
  if xy_null is not None:
    xy=np.concatenate((xy, xy_null),0)
    if null_zvalue is None:
      null_zvalue = z.max() + 0.1 * np.std(z)
    z =np.concatenate((z,np.repeat(null_zvalue, xy_null.shape[0])),0)
  if interp is None:
    interp = "nearest" # other options are "linear" and "cubic"
  # from http://stackoverflow.com/questions/15586919/inputs-x-and-y-must-be-1d-or-2d-error-in-matplotlib
  rangeX = max(xy[:,0]) - min(xy[:,0])
  rangeY = max(xy[:,1]) - min(xy[:,1])
  rmax = max(rangeX,rangeY)
  rangeX /= rmax
  rangeY /= rmax 
  xi = np.linspace(min(xy[:,0]), max(xy[:,0]), int(resolution * rangeX))
  yi = np.linspace(min(xy[:,1]), max(xy[:,1]), int(resolution * rangeY))
  X, Y = np.meshgrid(xi, yi)
  Z = interpolate.griddata((xy[:,0], xy[:,1]), np.array(z), (X, Y), method=interp)
  extent=[min(xy[:,0]),max(xy[:,0]),min(xy[:,1]),max(xy[:,1])]
  return Z, extent

def transparent_cmap(color=None, cmap=None, final_alpha=None):
  # http://stackoverflow.com/questions/10127284/overlay-imshow-plots-in-matplotlib
  if color is None:
    color = "blue"
  if (final_alpha is None) or (final_alpha < 0.01):
    final_alpha = 1.
  if cmap:
    mycmap = plt.get_cmap(cmap)
  else:
    from matplotlib.colors import colorConverter
    mycmap = matplotlib.colors.LinearSegmentedColormap.from_list('my_cmap',
        [colorConverter.to_rgba(color),colorConverter.to_rgba(color)],256)

  mycmap._init() # create the _lut array, with rgba values
  mycmap._lut[:,-1] = np.linspace(0, final_alpha, mycmap.N+3) # here it is progressive alpha array
  return mycmap

def pychem_mlr(x,y,order):
    """Multiple linear regression fit of the columns of matrix x  (dependent variables) to constituent vector y (independent variables)
    
    order -     order of a smoothing polynomial, which can be included in the set of independent variables. If order is
                not specified, no background will be included.
    b -         fit coeffs
    f -         fit result (m x 1 column vector)
    r -         residual   (m x 1 column vector)
    --          imported from PyChem http://pychem.sourceforge.net/ (Roger Jarvis, GPL)
    """
    if order > 0:
        s=np.ones((len(y),1))
        for j in range(order):
#           s=np.concatenate((s,(np.arange(0,1+(1.0/(len(y)-1)),1.0/(len(y)-1))**j)[:,np.newaxis]),1)
            s=np.concatenate((s,(np.linspace(0,1,len(y))**j)[:,np.newaxis]),1)
        X=np.concatenate((x, s),1)
    else:
        X = x
    #calc fit b=fit coefficients
    b = np.dot(np.dot(np.linalg.pinv(np.dot(np.transpose(X),X)),np.transpose(X)),y)
    f = np.dot(X,b)
    r = y - f

    return b,f,r

def find_minima (Y, alpha=None, baseline_width=None):
  """ 
  http://stackoverflow.com/questions/24656367/find-peaks-location-in-a-spectrum-numpy
  """
  if baseline_width is None:
    baseline_width = int(0.05 * len(Y))
  if not baseline_width%2:
    baseline_width += 1  # must be an odd number
  if alpha is None:
    alpha=0.05
  alpha *= np.std(Y)
  #Obtaining derivative
  dY = signal.convolve(Y, [1,0,-1], 'valid')
  #Checking for sign-flipping
  ddS = signal.convolve(np.sign(dY), [-1,0,1], 'valid')
  #These candidates are basically all negative slope positions (Add one since using 'valid' shrinks the arrays)
  candidates = np.where(dY < 0)[0] + 2
  #Here they are filtered on actually being the final such position in a run of negative slopes
  peaks = sorted(set(candidates).intersection(np.where(ddS == 2)[0] + 2))
  Ybase = signal.medfilt(Y, baseline_width) # baseline_width should be large in comparison to your peak X-axis lengths and an odd number.
  peaks = np.array(peaks)[Ybase[peaks] - Y[peaks] > alpha]
  return peaks

def find_min_max_convex_hull (x, y, exclude_max_border=None):
  if exclude_max_border is None:
    exclude_max_border=True
  hull=spatial.ConvexHull(np.append(x.reshape(-1,1),y.reshape(-1,1),1)) # N x 2 array (reshape since x and y can be vectors, N x 0)
  first_point=np.where(hull.vertices==0)[0][0] # if x is ordered, then first and last values should belong to C-Hull [1]
  # vertices are in counter-clockwise order, and are rotated s.t. pts starts at zero 
  pts=np.concatenate([hull.vertices[first_point:],hull.vertices[:first_point]]) 
  # [1] 1st and last points can however belong to min or max sets (if minima, neighboring points do not belong to C-Hull)
  if (pts[1] - pts[0] == 1): # then x[0] is a local max 
    if exclude_max_border:
      pts=pts[1:]
    else:
      pts=np.concatenate([pts[1:],pts[:1]]) # index of x[0] goes to end of list (will be split into list of max values)
  last=np.where(pts==(len(x)-1))[0][0] # last point from x
  if (pts[last] - pts[last-1] == 1): # then last is a local max; formally is a local max, but we may exclude it  
    minlist=np.sort(pts[:last]) # goes from 0 to (last-1)
    if exclude_max_border:
      maxlist=np.sort(pts[last+1:]) # excludes last element from list
    else:
      maxlist=np.sort(pts[last:]) #  includes last element in list
  else: # last is local minimum
    minlist=np.sort(pts[:last+1])
    maxlist=np.sort(pts[last+1:])
  return minlist, maxlist
 
def baseline_points_by_polynomial (x, y, order=None, noise_std=None, min_points=None, exclude=None, n_iter=None):
  """
    Find baseline points based on iterative polynomial fit (where only Y values lower than fitted are used in next
    iteration). Also returns the final fitted baseline. 
    Based on function spc.fit.poly.below() from hyperSpec library (@author Claudia Beleites)
    When exclude[] list of _indexes_ is given, these elements are prevented from belonging to baseline. 
  """
  if order is None:
    order = 5
  if n_iter is None: # original function in hyperSpec didn't have a limit
    n_iter =  len(x) # very large number (at each iteration it decreases at least one)
  if min_points is None:
    min_points = int (max(len(x)/20, 3 * order + 1)) # from spc.fit.poly.below() in hyperSpec
  if noise_std is None:
    noise_std = 0.1 # more strict since some Y values will be lower than fitted anyway
  noise = noise_std * np.std(y)
  new_valid=range(len(x))
  for counter in range(n_iter): # until x-axis of baseline becomes stable or consists of too few points 
    valid = list(new_valid) # deep copy (obs: deepcopy() would be needed for arbitrary objects) 
    if exclude:  
      valid = sorted(set(valid) - set(exclude)) # sorted list of valid indexes not in exclude[] list
    # polyfit will give a Vandermonde matrix (with coeffs only); poly1d is polyfunction itself;
    baseline=np.poly1d(np.polyfit(x[valid], y[valid], order))(x) # polyfunction calculated at all wavelengths
    # "list()" alternative to "np.array()" :  new_valid = np.where(y - baseline <= noise)[0] 
    new_valid = [i for i in range(len(x)) if (baseline[i] + noise >= y[i])]
    if (len(new_valid) < min_points) or (valid==new_valid):
          break # exits "while" statement
  return valid, baseline

def baseline_points_by_rubberband (x, y, band_size=None, noise_std=None):
  """
    Find baseline points through a piece-wise rubberband (convex hull): split time series into segments (where
    "band_size" is the fraction of points to be used per segment) and store local minima (lower part of convex hull).
    Also returns the "rubber bands", that is, the piecewise linear fits to the local minima. (Notice that the rubber
    bands may be a poor baseline without spline smoothing. They do not include all (noisy) baseline points, but just the
    minima...)
    Inspired by spc.rubberband() from hyperSpec library (@author Claudia Beleites)
  """
  if band_size is None or band_size < 10./float(len(x)):
    band_size = 1. # use all points -> a single "rubberband" around whole time series
  if noise_std is None:
    noise_std = 1. # less strict since baseline will be always smaller than Y values
  noise = noise_std * np.std(y)
  itv=[int(i) for i in np.linspace(0,len(x),1./band_size + 1)]
  valid = []
  for i in range(len(itv) - 1):
    minlist = find_min_max_convex_hull (x[itv[i]:itv[i+1]], y[itv[i]:itv[i+1]], exclude_max_border=True)[0] # first element of return list
    valid=np.concatenate([valid,minlist + itv[i]]) # minlist indexes refer to sliced arrays, therefore must sum with first element
  valid = sorted(set(valid))
  baseline = np.interp(x,x[valid],y[valid]) 
  pts = np.where(baseline + noise >= y)[0] 
  return pts, baseline

def baseline_points_by_als(x, y, smooth=None, asym=None, n_iter=None, exclude=None):
  """
    Find baseline points through "Asymmetric Least Squares Smoothing" (P. Eilers and H. Boelens, 2005).
    Returns also the fitted baseline.
    "smooth" is log(lambda) and "asym" is p in the paper (that is, lambda between 7 and 20 are equiv to 10^3 ~ 10^9
    as suggested in original paper)
    When exclude[] list of _indexes_ is given, these elements are prevented from belonging to baseline. 
  """
  if (smooth is None) or (smooth < 1):
    smooth = 10
  if (asym is None) or (asym > 0.999) or (asym < 1e-5):
    asym = 0.01
  if n_iter is None:
    n_iter = 5   # enough for finding points only; the baseline curve itself might need more iterations
  L = len(x)
  smooth = np.exp(smooth) 
  D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  w = np.ones(L)
  if exclude:
    w[exclude] = 0.
  for i in range(n_iter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + smooth * D.dot(D.transpose())
    z = sparse.linalg.spsolve(Z, w * y) # fitted baseline
    w = asym * (y > z) + (1. - asym) * (y < z)
    if exclude:
      w[exclude] = 0.
  pts = np.where(w > 0.9)[0] # w is the list of weights (usually very close to one or to zero)
  return pts, z

def baseline_subtraction(spectrum, baseline, zero=None):
  if zero is None:
    zero = False
  spectrum -= baseline
  if zero:
    spectrum -= min(spectrum) # force lowest value to zero
  else:
    spectrum -= min(spectrum) - min(baseline) # maintain original scale s.t. lowest value is baseline
  return spectrum 
    
def kneighbors_graph_by_idx (xy, idx, n_neighbors=None):
  """ 
    Returns a sparse matrix (CSR) with the kneighbors_graph, but using IDX array instead of XY indexes. Resulting sparse
    matrix may have connectivities larger than one (in contrast to sklearn.neighbors.kneighbors_graph which returns
    zeros and ones)
  """
  if n_neighbors is None:
    n_neighbors = 5
  # http://scikit-learn.org/stable/auto_examples/cluster/plot_ward_structured_vs_unstructured.html
  con_i = neighbors.kneighbors_graph(xy, n_neighbors=n_neighbors) # original indexes (XY positions)
  con_f = sparse.csr_matrix((max(idx)+1, max(idx)+1), dtype=np.int8) # ZERO sparse matrix w/ final indexes (from IDX array)
  for i in range(con_i.shape[0]):
    neighbor_idxs = idx[con_i.getrow(i).toarray()[0].nonzero()] # indexes of non-zero neighbors of i, mapped to IDX
    con_f[idx[i], neighbor_idxs] = 1
  return con_f

def median_polish(spc, max_iter=None):
  """
    Tukey's median polish alghoritm for additive models method - https://github.com/borisvish/Median-Polish 
  """
  if max_iter is None:
      max_iter = 5
  grand_effect = 0
  median_row_effects = 0
  median_col_effects = 0
  row_effects = np.zeros(shape=spc.shape[0])
  col_effects = np.zeros(shape=spc.shape[1])
  for i in range(max_iter):
    row_medians = np.median(spc,1) 
    row_effects += row_medians
    median_row_effects = np.median(row_effects)
    grand_effect += median_row_effects
    row_effects -= median_row_effects
    spc -= row_medians[:,np.newaxis] 

    col_medians = np.median(spc,0) 
    col_effects += col_medians
    median_col_effects = np.median(col_effects)

    spc -= col_medians 
    grand_effect += median_col_effects

  return spc, row_effects, col_effects 

def residual_around_wl(x, y, x_val, interpolation=None, n_exclude=None, smoothspline=None, smoothdata=None, debug=None):
  """
    Predicts the intensity at x value x_val based on interpolation, after removing observed intensities around
    x_val. Returns residual given as the difference between observed and predicted values (exactly) at x_val, where the
    exact observed value is found by linear interpolation if absent. Possible interpolation algorithms for flanking 
    areas are 'als' (baseline points by assymetric linear square), 'poly' (polynomial interpolation over iterative baseline
    points), 'spline' (cubic spline interpolation over all surrounding points), 'wspline' (spline interpolation
    using all points, but weighted s.t. more distant x values have more influence). 
    
    Notice that 'als'and 'poly' will search and use only baseline (lowest) points from flanking areas (that is, after 
    excluding points close to x_val). Notice also that 'wspline' uses smoothing controled by 'smooth' variable, while 
    'spline' doesn't. Weights are applied only to "excluded" area in 'wspline', and therefore you can chose to 'exclude' 
    more points than for other methods. 

    Vectors x and y can be smoothed out before any interpolation proceeds, and "debug" boolean will output several
    variables, namely the included and excluded indexes in the interpolation, as well as the expected and observed
    intensities separatedly. x_val can be a list.
  """
# first thing, since we may change x and y by smoothed versions
  x_val = np.array(x_val)
  observed = np.interp(x_val, x, y) # in case x_val is not exactly on list x
  if smoothdata: # work with smoothed version of data
    if smoothdata > len(x)/3:
      smoothdata = len(x)/3
    x = moving_average (x, n = smoothdata, power=0.1) # True = 1, but user can give other values
    y = moving_average (y, n = smoothdata, power=0.1)
  if interpolation is None:
    interpolation = "spline"
  if n_exclude is None: # excludes 2 x n_exclude + 1 (x_val and n_exclude neighbours on each side)
    n_exclude = 1
  if interpolation is not "wspline" and n_exclude > int((len(x) - 3)/6): # removed points should be < 1/3 of series 
    n_exclude = int((len(x) - 3)/6)
  if smoothspline is None or smoothspline > 0.5:
    smoothspline = 0.001 # for weighted smoothed splines
  # idx, b = min(enumerate(a),  key=lambda x: abs(x[1]-val)) # returns index and element of "a" closest to "val"
  if x_val.shape:
    idx_min = min(range(len(x)), key=lambda i: abs(x[i]-x_val.min())) # index of x closest to x_val
    idx_max = min(range(len(x)), key=lambda i: abs(x[i]-x_val.max())) 
  else:
    idx_min = idx_max = min(range(len(x)), key=lambda i: abs(x[i]-x_val)) # index of x closest to x_val
  i_idx = np.max([idx_min - n_exclude, 0]) # leftmost element to remove
  f_idx = np.min([idx_max + n_exclude + 1, len(x)]) # rightmost idx to be excluded

  if interpolation == "als":
    valid, fit = baseline_points_by_als(x, y, n_iter=1, smooth=5, exclude=range(i_idx,f_idx))
    estimated = np.interp(x_val, x, fit) # in case x_val is not exactly on list x

  elif interpolation == "poly":
    valid, fit = baseline_points_by_polynomial(x, y, n_iter = 2, exclude=range(i_idx,f_idx))
    estimated = np.interp(x_val, x, fit) # in case x_val is not exactly on list x

  elif interpolation == "spline":
    valid=np.concatenate((  range(0,i_idx), range(f_idx+1,len(x))  )).astype(int) # concatenate() returns as floats
    estimated = interpolate.InterpolatedUnivariateSpline(x[valid], y[valid])(x_val) # spline OO function
    if debug: # usually not needed; only for debbuging (plotting) purposes
      fit = interpolate.InterpolatedUnivariateSpline(x[valid], y[valid])(x) # spline OO function

  elif interpolation == "wspline":
    w = np.absolute(x[range(i_idx,f_idx)] - x_val) # distance from x_val 
    w = smoothspline + (1.-smoothspline) * (w - np.min(w))/(np.max(w) - np.min(w))
    weights = np.ones(len(x))
    weights[range(i_idx,f_idx)] = w
    estimated = interpolate.splev(x_val, interpolate.splrep (x, y, w=weights, s=2*smoothspline), der=0)
    if debug: # "valid" is not used by this function, but we have to report it for consistency
      valid=np.concatenate((  range(0,i_idx), range(f_idx+1,len(x))  )).astype(int) # concatenate() returns as floats
      fit = interpolate.splev(x, interpolate.splrep (x, y, w=weights, s=2*smoothspline), der=0)
  
  if debug: # observed value; expected value; valid (included) points; excluded points; x; y; estimated y values for all  x
    return observed, estimated, valid, range(i_idx,f_idx), x, y, fit
  else:
    return observed - estimated

# multiprocessing.pool functions
def optim_nnls_parallel (params): 
  return np.array([optimize.nnls(params[0], params[1][i])[0] for i in range(len(params[1]))])


def find_vca_indexes(specmat, n=None, approximate=None):
  """ https://github.com/Chathurga/unmixR vcaMod.R"""
  if n is None:
    n = 4
  if approximate is None or approximate is False:
    Y = decomposition.PCA(n_components=n).fit_transform(specmat).T # n rows and nsamples cols
  else: 
    Y = decomposition.RandomizedPCA(n_components=n).fit_transform(specmat).T
  E = np.zeros((n,n+1),dtype=float)
  U = np.zeros((n,n),dtype=float)
  E[n-1,0] = 1.
  w = np.ones(n,dtype=float)
  proj_acc = np.zeros(n,dtype=float)
  indices = np.zeros(n,dtype=int) - 1 # minus one for debugging
  for i in range(n):
    U[:,i] = E[:,i]
    for j in range(2,n):
      proj_e_u = (np.dot(E[:,i],U[:,j-1])/np.dot(U[:,j-1],U[:,j-1])) *  U[:,j-1]
      U[:,i] -= proj_e_u
    proj_w_u = (np.dot(w,U[:,i])/np.dot(U[:,i],U[:,i])) * U[:,i]
    proj_acc += proj_w_u
    f  = w - proj_acc
    if i == 0:
      proj_acc = np.zeros(n,dtype=float)
    v = np.dot(f,Y)
    idx = v.argmax()
    indices[i] = idx
    E[:,i+1] = Y[:,idx]

  return indices

def find_vca_lopez(specmat, n=None, approximate=None):
  """ https://github.com/Chathurga/unmixR vcaLopez.R"""
  if n is None:
    n = 4
  if approximate is None or approximate is False:
    Y = decomposition.PCA(n_components=n).fit_transform(specmat).T # n rows and nsamples cols
  else: 
    Y = decomposition.RandomizedPCA(n_components=n).fit_transform(specmat).T
  E = np.zeros((n,n),dtype=float)
  E[n-1,0]=1
  I = np.eye(n) 
  indices = np.zeros(n,dtype=int) - 1 # for debugging (should not be negative)
  for i in range(n):
    w = np.random.random(n)
    x = np.dot(I - E * np.linalg.pinv(E), w) # dot(mat,vec)=vec BUT mat*vec=mat
    f = x/np.sqrt(np.power(x,2).sum())
    v = np.dot(f,Y)
    print ("E=",E, "\n v=",v)
    idx = v.argmax()
    indices[i] = idx
    E[:,i] = Y[:,idx]

  return indices

def distance_SID(s1, s2):
  """
  Computes the spectral information divergence between two vectors.
  Reference : C.-I. Chang, "An Information-Theoretic Approach to SpectralVariability,
      Similarity, and Discrimination for Hyperspectral Image" IEEE TRANSACTIONS ON 
      INFORMATION THEORY, VOL. 46, NO. 5, AUGUST 2000.
    --> imported from pysptools
  """
  p = (s1 / np.sum(s1)) + np.spacing(1)
  q = (s2 / np.sum(s2)) + np.spacing(1)
  return np.sum(p * np.log(p / q) + q * np.log(q / p))

def distance_SAM(s1, s2):
  """
  Computes the spectral angle mapper between two vectors (in radians).
    --> imported from pysptools
  """
  try:
    s1_norm = np.sqrt(np.dot(s1, s1))
    s2_norm = np.sqrt(np.dot(s2, s2))
    sum_s1_s2 = np.dot(s1, s2)
    angle = np.arccos(sum_s1_s2 / (s1_norm * s2_norm))
  except ValueError: # python math don't like when acos is called with a value very near to 1
    return 0.0
  return angle

def distance_crosscorr(s1, s2):
  """
  Computes the normalized cross correlation distance between two vectors.
  Dist is between [-1, 1], with one indicating a perfect match.
    --> imported from pysptools
  """
  s = s1.shape[0]
  corr = np.sum((s1 - np.mean(s1)) * (s2 - np.mean(s2))) / (stats.tstd(s1) * stats.tstd(s2))
  return corr * (1./(s-1))

def unconstrained_MCR (specs, n_members=None, max_iter=None, threshold=None, normalize=None, closure=None):
    if n_members is None:
        n_members = 4
    if max_iter is None:
        max_iter = 100
    if closure is None:
        closure = False
    if threshold is None:
        threshold = 0.001
    if normalize is None:
        normalize = False
    from pysptools.eea import nfindr
    # initial estimates, by NFINDR and non-negative LR (for abundances only)
    endmembers, _, _, _ = nfindr.NFINDR (specs, n_members)
    if normalize is True:
        endmembers = np.array([tspec/np.sum(tspec) for tspec in endmembers],dtype=float)
    abundances = np.dot(specs, np.linalg.pinv(endmembers.T).T)
    if closure is True:
        abundances = np.array([(i/np.sum(i)) for i in abundances])
    resid = np.array([specs[i,:] - np.dot(abundances[i,:], endmembers) for i in range(specs.shape[0])])
    initial_rss = old_rss = np.sum(np.power(resid,2)) # residual sum of squares
    for iteration in range(max_iter):
        endmembers = np.dot(specs.T, np.linalg.pinv(abundances.T)).T
        if normalize is True:
            endmembers = np.array([tspec/np.sum(tspec) for tspec in endmembers],dtype=float)
        abundances = np.dot(specs, np.linalg.pinv(endmembers.T).T)
        if closure is True:
            abundances = np.array([(i/np.sum(i)) for i in abundances])
        resid = np.array([specs[i,:] - np.dot(abundances[i,:], endmembers) for i in range(specs.shape[0])])
        rss = np.sum(np.power(resid,2)) # residual sum of squares
        if (old_rss - rss)/old_rss < threshold:
            break # finished
        old_rss = rss
    sys.stdout.write("finished in " + str(iteration) + " iterations\n");
    ordered_idx = np.argsort(np.max(endmembers,1))[::-1] # from largest to lowest
    endmembers = endmembers[ordered_idx] # reorder rows s.t. first endmembers have highest intensities
    abundances = abundances[:,ordered_idx] # reorder columns of abundances to follow order of endmembers
    rss = np.array([np.sum(r)/endmembers.shape[1] for r in np.power(resid,2)]).reshape(1,-1) # RSS per spectrum
    return abundances, endmembers, rss

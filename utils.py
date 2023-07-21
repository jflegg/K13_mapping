from pylab import *
from numpy import *
import pymc as pm
import map_utils
import os
import hashlib
from osgeo import gdal

# ================================================================================
#  Find "closest" nodes
# ================================================================================
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return where(dist_2==min(dist_2))[0]

# ================================================================================
#  Create unique list of space time locations 
# ================================================================================
def Unique_Sonly_points(disttol, data_mesh):

    # Find near spatiotemporal duplicates.
    ui = [0]
    fi = [0]
    dx = np.empty(1)
    for i in range(1, data_mesh.shape[0]):
        match=False
        for j in range(len(ui)):
            pm.gp.distances.geographic(dx, data_mesh[i,:2].reshape((1,2)), data_mesh[ui[j],:2].reshape((1,2)))
            if dx[0]<=disttol:
                match=True
                fi.append(j)
                break
        if not match:
            ui.append(i)
            fi.append(max(fi)+1)

    ui = np.array(ui)
    fi = np.array(fi)
    fi = np.array(fi)

    return ui, fi


# ================================================================================
#  Create unique list of space time locations 
# ================================================================================
def Unique_ST_points(disttol, ttol, data_mesh):

    # Find near spatiotemporal duplicates.
    ui = [0]
    fi = [0]
    dx = np.empty(1)
    for i in range(1, data_mesh.shape[0]):
        match=False
        for j in range(len(ui)):
            pm.gp.distances.geographic(dx, data_mesh[i,:2].reshape((1,2)), data_mesh[ui[j],:2].reshape((1,2)))
            dt = abs(data_mesh[ui[j],2]-data_mesh[i,2])
            if dx[0]<=disttol and dt<=ttol:
                match=True
                fi.append(j)
                break
        if not match:
            ui.append(i)
            fi.append(max(fi)+1)

    ui = np.array(ui)
    fi = np.array(fi)
    ti = [np.where(fi == i)[0] for i in range(max(fi)+1)]
    fi = np.array(fi)

    return ui, fi, ti


# ================================================================================
#  Create unique list of space time locations, for main, lower and upper datasets
# ================================================================================
def Unique_ST_points_main_upper_lower(data_mesh_main, data_mesh_lower, data_mesh_upper, disttol, ttol):
    
    # Find which spatiotemporal points in upper are duplicates of those in main
    ui = []
    dx = np.empty(1)
    for i in range(0, data_mesh_upper.shape[0]):
        match=False
        for j in range(data_mesh_main.shape[0]):
            pm.gp.distances.geographic(dx, data_mesh_upper[i,:2].reshape((1,2)), data_mesh_main[j,:2].reshape((1,2)))
            dt = abs(data_mesh_main[j,2]-data_mesh_upper[i,2])
            if dx[0]<=disttol and dt<=ttol:
                match=True
                break
        if not match:
            ui.append(i)

    ui_upper = np.array(ui)
    # Get the distinct spatiotemporal points in upper
    if len(ui_upper)>0:
        mesh_upper = data_mesh_upper[ui_upper,:]
        n_upper = mesh_upper.shape[0]
    elif len(ui_upper)==0:
        mesh_upper = []
        n_upper = 0

    # Find which spatiotemporal points in lower are duplicates of those in main
    ui = []
    dx = np.empty(1)
    for i in range(0, data_mesh_lower.shape[0]):
        match=False
        for j in range(data_mesh_main.shape[0]):
            pm.gp.distances.geographic(dx, data_mesh_lower[i,:2].reshape((1,2)), data_mesh_main[j,:2].reshape((1,2)))
            dt = abs(data_mesh_main[j,2]-data_mesh_lower[i,2])
            if dx[0]<=disttol and dt<=ttol:
                match=True
                break
        if not match:
            ui.append(i)

    ui_lower = np.array(ui)
    # Get the distinct spatiotemporal points in lower
    if len(ui_lower)>0:
        mesh_lower = data_mesh_lower[ui_lower,:]
        n_lower = mesh_lower.shape[0]
    elif len(ui_lower)==0:
        mesh_lower = []
        n_lower = 0

    n_main = data_mesh_main.shape[0]

    if (n_lower == 0) & (n_upper == 0):
        mesh_combined = vstack((data_mesh_main))
    elif (n_lower == 0):
        mesh_combined = vstack((data_mesh_main, mesh_upper))
    elif (n_upper == 0):
        mesh_combined = vstack((data_mesh_main, mesh_lower))
    else:
        mesh_combined = vstack((data_mesh_main, mesh_upper,mesh_lower))

    return n_main, n_lower, n_upper, mesh_combined, ui_upper, ui_lower


# ================================================================================
#  Covert lon and lat from radians to degrees or vice-versa
# ================================================================================
def convert_coords(x,y, frm, to):
    """Converts longitude (x) and latitude (y) from 'frm' (degrees or radians) to 'to' (radians or degrees)a grid to a new"""

    # convert from degrees to radians
    if (frm == 'degrees') & (to == 'radians'):
        if (all(x <= pi/2.)) & (all(y <= pi)) & (all(y >= -pi)):
            print('already in radians')
        else:
            x = x*pi/180.
            y = y*pi/180.
    # convert from radians to degrees
    elif (frm == 'radians') & (to == 'degrees'):
        if any(x > pi/2.) & all(y > pi) & all(y < -pi):
            print('already in degrees')
        else:
            x = x*180./pi
            y = y*180./pi
    else:
        print('something is wrong with this request to convert coordinates!')
    
    return x, y

# ================================================================================
#  Extract interpolated values based on an input raster
# ================================================================================
def extract_environment(layer_name, x, postproc=lambda x:x, id_=None, lock=None):
    "Expects ALL locations to be in decimal degrees."
    
    fname = hashlib.sha1(x.tostring()+layer_name+str(id_)).hexdigest()+'.npy'
    path, name = os.path.split(layer_name)
    name = os.path.splitext(name)[0]
    if fname in os.listdir(path):
        return name, np.load(os.path.join(path,fname))
    else:    
        grid_lon, grid_lat, grid_data, grid_type = map_utils.import_raster(name,path, type=None)
        # Convert to centroids
        grid_lon += (grid_lon[1]-grid_lon[0])/2.
        grid_lat += (grid_lat[1]-grid_lat[0])/2.
        # Interpolate
        extracted = map_utils.interp_geodata(grid_lon, grid_lat, postproc(grid_data).data, x[:,0], x[:,1], grid_data.mask, chunk=None, view='y-x+', order=0)
        del grid_data
        np.save(os.path.join(path,fname), extracted)
        return name, extracted

def validate_format_str(st):
    for i in [0,2]:
        if not st[i] in ['x','y']:
            raise ValueError('Directions must be x or y')
    for j in [1,3]:
        if not st[j] in ['-', '+']:
            raise ValueError('Orders must be + or -')
            
    if st[0]==st[2]:
        raise ValueError('Directions must be different')
    
# ================================================================================
#  Covert grid to a new layout
# ================================================================================
def grid_convert(g, frm, to, validate=False):
    """Converts a grid to a new layout.
      - g : 2d array
      - frm : format string
      - to : format string

      Example format strings:
        - x+y+ (the way Anand does it) means that 
            - g[i+1,j] is west of g[i,j]
            - g[i,j+1] is north of g[i,j]
        - y-x+ (map view) means that 
            - g[i+1,j] is south of g[i,j]
            - g[i,j+1] is west of g[i,j]"""

    # Validate format strings
    if validate:
        for st in [frm, to]:
            validate_format_str(st)

    # Transpose if necessary
    if not frm[0]==to[0]:
        g = g.T

    first_dir = to[1]
    if not first_dir == frm[frm.find(to[0])+1]:
        g=g[::-1,:]

    sec_dir = to[3]
    if not sec_dir == frm[frm.find(to[2])+1]:
        g=g[:,::-1]

    # print first_dir, sec_dir
    return g



# ================================================================================
#    Get covariates given the lon, lat, year for pf (2000-2019)
# ================================================================================
def getCovariatesForLocations2019(data_mesh): 

    N = data_mesh.shape[0]
    pf = np.zeros(N)
    for i in range(N):
        x = data_mesh[i,0:2]*180./pi # convert to degrees
        year = int(data_mesh[i,2])
        #if year < 2000:
        #    year = 2000
        #elif year > 2017: 
        #    year = 2017
        # pf only in Africa, 2000 - 2017    
        #raster = gdal.Open('/Users/jflegg/data/MAP_pf_Africa_2000_2015/MODEL43.%s'%year+'.PR.rmean.stable.tif')
        #raster = gdal.Open('/Users/jflegg/data/2019_Global_PfPR/2019_Global_PfPR_%s'%year+'.tif')
        # pf only in Africa, 2000 - 2019
        if year < 2000:
            year = 2000
        elif year > 2019: 
            year = 2019
        raster = gdal.Open('/Users/jflegg/data/2020_PfPR/Raster Data/PfPR_median/PfPR_median_Global_admin0_%s'%year+'.tif')    
        myarray = array(raster.GetRasterBand(1).ReadAsArray())
        masked_array =np.ma.masked_where(myarray ==myarray[1][1], myarray)
        masked_array.data[where(masked_array.data==myarray[1][1])] = nan
        data_yp_xp = map_utils.grid_convert(ma.masked_array(masked_array.data, mask = masked_array.mask), 'y-x+','y+x+', validate = True)
        width = raster.RasterXSize
        height = raster.RasterYSize
        gt = raster.GetGeoTransform()
        minx = gt[0]
        maxy = gt[3] 
        dx =  gt[1]
        dy = gt[5]
        miny = maxy + (height-1)*dy
        maxx = minx + (width-1)*dx 
        xplot = arange(minx, maxx+dx/2, dx)
        yplot = arange(miny, maxy-dy/2, -dy)
        xint = int((x[0] - gt[0])/gt[1]) 
        yint = int((x[1] - gt[3])/(-gt[5])) 
        # only consider close points where sutiability is not nan and where nonmasked 
        ind_to_use_close_points = np.where(1-data_yp_xp.mask.astype(np.float32))
        # use average of neighbouring points, else find the closest non-nan value
        if len(where(isnan(data_yp_xp.data[yint-1:yint+2, xint-1:xint+2])==False)[0])==0:
            node = array([x[0], x[1]]) 
            nodes = zeros([len(ind_to_use_close_points[0]),4])
            nodes[:,0]= xplot[ind_to_use_close_points[1]]
            nodes[:,1]= yplot[ind_to_use_close_points[0]]
            nodes[:,2]= ind_to_use_close_points[1]            
            nodes[:,3]= ind_to_use_close_points[0]
            close_inds = closest_node(node, nodes[:,0:2]) # find closest non nan node point and take that value
            pf[i] = data_yp_xp.data[int(nodes[close_inds,3][0]), int(nodes[close_inds,2][0])]
        else:
            pf[i] = nanmean(data_yp_xp.data[yint-1:yint+2, xint-1:xint+2])

    return pf
    


# ================================================================================
#    Get covariates given the lon, lat, year for pf (2000-2020)
# ================================================================================
def getCovariatesForLocations(data_mesh): 

    N = data_mesh.shape[0]
    pf = np.zeros(N)
    for i in range(N):
        x = data_mesh[i,0:2]*180./pi # convert to degrees
        year = int(data_mesh[i,2])
        if year < 2000:
            year = 2000
        elif year > 2020: 
            year = 2020
     
        raster = gdal.Open('/Users/jflegg/data/202206_Global_Pf_Parasite_Rate_2000/202206_Global_Pf_Parasite_Rate_%s'%year+'.tif')    
        myarray = array(raster.GetRasterBand(1).ReadAsArray())
        masked_array =np.ma.masked_where(myarray ==myarray[1][1], myarray)
        masked_array.data[where(masked_array.data==myarray[1][1])] = nan
        data_yp_xp = map_utils.grid_convert(ma.masked_array(masked_array.data, mask = masked_array.mask), 'y-x+','y+x+', validate = True)
        width = raster.RasterXSize
        height = raster.RasterYSize
        gt = raster.GetGeoTransform()
        minx = gt[0]
        maxy = gt[3] 
        dx =  gt[1]
        dy = gt[5]
        miny = maxy + (height-1)*dy
        maxx = minx + (width-1)*dx 
        xplot = arange(minx, maxx+dx/2, dx)
        yplot = arange(miny, maxy-dy/2, -dy)
        xint = int((x[0] - gt[0])/gt[1]) 
        yint = int((x[1] - gt[3])/(-gt[5])) 
        # only consider close points where sutiability is not nan and where nonmasked 
        ind_to_use_close_points = np.where(1-data_yp_xp.mask.astype(np.float32))
        # use average of neighbouring points, else find the closest non-nan value
        if len(where(isnan(data_yp_xp.data[yint-1:yint+2, xint-1:xint+2])==False)[0])==0:
            node = array([x[0], x[1]]) 
            nodes = zeros([len(ind_to_use_close_points[0]),4])
            nodes[:,0]= xplot[ind_to_use_close_points[1]]
            nodes[:,1]= yplot[ind_to_use_close_points[0]]
            nodes[:,2]= ind_to_use_close_points[1]            
            nodes[:,3]= ind_to_use_close_points[0]
            close_inds = closest_node(node, nodes[:,0:2]) # find closest non nan node point and take that value
            pf[i] = data_yp_xp.data[int(nodes[close_inds,3][0]), int(nodes[close_inds,2][0])]
        else:
            pf[i] = nanmean(data_yp_xp.data[yint-1:yint+2, xint-1:xint+2])

    return pf
    


# ================================================================================
#  Get covariates given the lon, lat, year for suitability and pf (2000-2020)
# ================================================================================
def getCovariatesForLocationsK13maps(data_mesh): 

    path_name_utils = '/Users/jflegg'
    # pf data extraction
    N = data_mesh.shape[0]
    pf = np.zeros(N)
    for i in range(N):
        x = data_mesh[i,0:2]*180./pi # convert to degrees
        year = int(data_mesh[i,2])
        if year < 2000:
            year = 2000
        elif year > 2020: 
            year = 2020
        raster = gdal.Open(path_name_utils+'/data/202206_Global_Pf_Parasite_Rate_2000/202206_Global_Pf_Parasite_Rate_%s'%year+'.tif')   
        myarray = array(raster.GetRasterBand(1).ReadAsArray())
        masked_array =np.ma.masked_where(myarray ==myarray[1][1], myarray)
        masked_array.data[where(masked_array.data==myarray[1][1])] = nan
        data_yp_xp = map_utils.grid_convert(ma.masked_array(masked_array.data, mask = masked_array.mask), 'y-x+','y+x+', validate = True)
        gt = raster.GetGeoTransform()
        xint = int((x[0] - gt[0])/gt[1]) 
        yint = int((x[1] - gt[3])/(-gt[5])) 
        pf[i] = nanmean(data_yp_xp.data[yint-1:yint+2, xint-1:xint+2])

    # suitability data extraction
    raster = gdal.Open(path_name_utils+'/data/temperature_suitability/2010_TempSuitability.Pf.Index.1k.global_Decompressed.geotiff')
    myarray = array(raster.GetRasterBand(1).ReadAsArray())
    masked_array =np.ma.masked_where(myarray ==myarray[1][1], myarray)
    masked_array.data[where(masked_array.data==myarray[1][1])] = nan
    data_yp_xp = map_utils.grid_convert(ma.masked_array(masked_array.data, mask = masked_array.mask), 'y-x+','y+x+', validate = True)
    gt = raster.GetGeoTransform()
    suitability = np.zeros(N)
    for i in range(N):
        x = data_mesh[i,0:2]*180./pi # convert to degrees
        xint = int((x[0] - gt[0])/gt[1]) 
        yint = int((x[1] - gt[3])/(-gt[5])) 
        suitability[i] = nanmean(data_yp_xp.data[yint-1:yint+2, xint-1:xint+2])

    # population data extraction
    raster = gdal.Open(path_name_utils+'/data/WorldPop/ppp_2020_1km_Aggregated.tif')
    myarray = array(raster.GetRasterBand(1).ReadAsArray())
    masked_array =np.ma.masked_where(myarray ==myarray[1][1], myarray)
    masked_array.data[where(masked_array.data==myarray[1][1])] = nan
    data_yp_xp = map_utils.grid_convert(ma.masked_array(masked_array.data, mask = masked_array.mask), 'y-x+','y+x+', validate = True)
    gt = raster.GetGeoTransform()
    population = np.zeros(N)
    for i in range(N):
        x = data_mesh[i,0:2]*180./pi # convert to degrees
        xint = int((x[0] - gt[0])/gt[1]) 
        yint = int((x[1] - gt[3])/(-gt[5])) 
        population[i] = nanmean(data_yp_xp.data[yint-1:yint+2, xint-1:xint+2])

    # travel time data extraction (2015)
    raster = gdal.Open(path_name_utils+'/data/2015_accessibility_to_cities/2015_accessibility_to_cities_v1.0.tif')
    myarray = array(raster.GetRasterBand(1).ReadAsArray())
    masked_array =np.ma.masked_where(myarray ==myarray[1][1], myarray)
    data_temp = masked_array.data # need an extra step to convert from integer to float
    data_temp = data_temp.astype(float)
    data_temp[where(masked_array.data==myarray[1][1])] = nan
    data_yp_xp = map_utils.grid_convert(ma.masked_array(data_temp, mask = masked_array.mask), 'y-x+','y+x+', validate = True)
    gt = raster.GetGeoTransform()
    traveltime2015 = np.zeros(N)
    for i in range(N):
        x = data_mesh[i,0:2]*180./pi # convert to degrees
        xint = int((x[0] - gt[0])/gt[1]) 
        yint = int((x[1] - gt[3])/(-gt[5])) 
        traveltime2015[i] = nanmean(data_yp_xp.data[yint-1:yint+2, xint-1:xint+2])    

    # travel time data extraction (2000) - not that this is NOT returned
    xplot_accessibility, yplot_accessibility, data_accessibility, TypeRegion = map_utils.import_raster('access100001',path_name_utils+'/data/trav-time-100k', type = None) # these are in radians
    data_yp_xp = ma.masked_array(map_utils.grid_convert(data_accessibility.data, 'y-x+','y+x+', validate = True), mask=map_utils.grid_convert(data_accessibility.mask , 'y-x+','y+x+', validate = True))
    dx = xplot_accessibility[1] - xplot_accessibility[0]
    dy = yplot_accessibility[1] - yplot_accessibility[0]
    traveltime2000 = np.zeros(N)
    for i in range(N):
        x = data_mesh[i,0:2]*180./pi # convert to degrees
        xint = int((x[0] - xplot_accessibility[0])/dx) 
        yint = int((x[1] - yplot_accessibility[-1])/dy) 
        traveltime2000[i] = nanmean(data_yp_xp.data[yint-1:yint+2, xint-1:xint+2])

    return pf, suitability, population, traveltime2015


def csv2rec(filename):
    return recfromtxt(filename, dtype=None, delimiter=',', names=True, encoding='utf-8')



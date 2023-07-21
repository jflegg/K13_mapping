from model import *
from pylab import *
from numpy import *
import pymc as pm
import map_utils
import model
import imp
imp.reload(model)
import math
from osgeo import gdal
import shapefile
import pdb
path_name_plot_K13 = '/Users/'


def plot_slices(data, name, first_year, thin, with_covariates, continent = "Africa"):

    
    # ================================================================================
    # Get a 'name' for saving the images
    # ================================================================================
    name_temp = os.path.split(name)[1][0:3]
    save_rasters_all = 1
    final_year = 2023

    # ================================================================================
    # Load the database and model
    # ================================================================================
    prev_db = pm.database.hdf5.load('res_db.hdf5')
    ResistanceSampler = pm.MCMC(model.make_model(**data),db=prev_db,dbmode='a', dbcomplevel=1,dbcomplib='zlib',dbname='res_db.hdf5')


    # ================================================================================
    # Countries borders (plotting)
    # ================================================================================
    sf = shapefile.Reader("/Users/jflegg/data/country borders/cn.dbf")
    records    = sf.records()
    shapes  = sf.shapes()
    Nshp    = len(shapes)
    ToPlot = array(["Thailand", "Myanmar","Cambodia","Viet Nam", "Lao People's Democratic Republic", "China", "Malaysia","Bangladesh", "Indonesia"])


    # ================================================================================
    # Get the lon, lat values across continent
    # ================================================================================
    xplot, yplot, Data, Type = map_utils.import_raster('AsiaRegionMekong','/Users/jflegg/data/flt and hdr files for mapping codes', type = None) # these are in radians
    data_yp_xp = map_utils.grid_convert(ma.masked_array(Data.data, mask = Data.mask), 'y-x+','y+x+', validate = True)
    where_unmasked2 = np.where(1-data_yp_xp.mask.astype(np.float32))

 
    # ================================================================================
    # Set up the lon, lat, year to do the slice plotting 
    # ================================================================================
    dplot2 = dstack(meshgrid(xplot,yplot))[where_unmasked2]
    dplot2_temp = zeros([dplot2.shape[0],3])
    dplot2_temp[:,0]= dplot2[:,0]
    dplot2_temp[:,1]= dplot2[:,1]


    # ================================================================================
    # Create the right directories if they don't already exist
    # ================================================================================
    my_fig_path = 'figures/'
    if not os.path.isdir(my_fig_path):
        os.makedirs(my_fig_path)
    mypath = 'rasters/'
    if not os.path.isdir(mypath):
        os.makedirs(mypath) 

    
    # ================================================================================
    # Get the suitability covariate, load suitability raster
    # ================================================================================
    if with_covariates:
        xplot_temp, yplot_temp, Data_suitability, Type = map_utils.import_raster('TemperatureSuitability','/Users/jflegg/data/temperature_suitability', type = None) 
        suitability_data = map_utils.grid_convert(Data_suitability.data, 'y-x+','y+x+', validate = True)[where_unmasked2]
        xplot_temp, yplot_temp, Data_population, Type = map_utils.import_raster('Population','/Users/jflegg/data/WorldPop', type = None) 
        population_data = map_utils.grid_convert(Data_population.data, 'y-x+','y+x+', validate = True)[where_unmasked2]
        #xplot_temp, yplot_temp, Data_traveltime, Type = map_utils.import_raster('Accessibility','/Users/jflegg/data/trav-time-100k', type = None) 
        xplot_temp, yplot_temp, Data_traveltime, Type = map_utils.import_raster('Accessibility2015','/Users/jflegg/data/2015_accessibility_to_cities', type = None) 
        Data_traveltime_temp = Data_traveltime.data # need an extra step to convert from integer to float
        Data_traveltime_temp = Data_traveltime_temp.astype(float)
        traveltime_data = map_utils.grid_convert(Data_traveltime_temp, 'y-x+','y+x+', validate = True)[where_unmasked2]


    # ================================================================================
    # How many saved traces there are:
    # ================================================================================
    n = len(ResistanceSampler.trace('Vinv')[:])
    temp_cutoffs_30 = zeros([int(math.ceil(float(n)/thin)), len(range(first_year,final_year))])


    # ================================================================================
    # Loop over each of the years to plot
    # ================================================================================
    for j in range(first_year,final_year):
        print('Plotting for t = %s'%j)
        times = j*np.ones(dplot2_temp.shape[0])
        dplot2_temp[:,2]=times


        # ================================================================================
        # Initialise the matrices for storing the output
        # ================================================================================
        Msurf = zeros(data_yp_xp.shape)
        Mean_temp = zeros(data_yp_xp.shape)
        E2surf = zeros(data_yp_xp.shape)
        E_temp = zeros(data_yp_xp.shape)
        temp = zeros([data_yp_xp.shape[0], data_yp_xp.shape[1],int(math.ceil(float(n)/thin))])


        # ================================================================================
        # Get the pf covariate for the right year 
        # ================================================================================
        if with_covariates:
            jtemp = j
            if j < 2000:
                jtemp = 2000
            elif j > 2020:
                jtemp = 2020
            if continent == "Africa":
                xplot_temp, yplot_temp, Data_pf, Type = map_utils.import_raster('Africa_pfpr_%s'%jtemp, path_name_plot_K13 +'/data/202206_Global_Pf_Parasite_Rate_2000', type = None) 
            elif continent == "India":
                xplot_temp, yplot_temp, Data_pf, Type = map_utils.import_raster('India_pfpr_%s'%jtemp, path_name_plot_K13 +'/data/202206_Global_Pf_Parasite_Rate_2000', type = None) 
            elif continent == "Mekong":
                xplot_temp, yplot_temp, Data_pf, Type = map_utils.import_raster('Mekong_pfpr_%s'%jtemp, path_name_plot_K13 +'/data/202206_Global_Pf_Parasite_Rate_2000', type = None) 
            pf_year_data = map_utils.grid_convert(Data_pf.data, 'y-x+','y+x+', validate = True)[where_unmasked2]


        # ================================================================================
        # Get E[v] and E[v**2] over the entire posterior, working in y+x+ orientation
        # ================================================================================
        count = 0
        for i in range(0,n,thin):
            # Reset all variables to their values at frame i of the trace
            ResistanceSampler.remember(0,i)
            Msurf_temp, Vsurf_temp = pm.gp.point_eval(ResistanceSampler.S.M_obs.value,ResistanceSampler.S.C_obs.value, dplot2_temp)
            if with_covariates:
                Msurf_temp +=  ResistanceSampler.trace('beta_1')[i]*(j-2000) + ResistanceSampler.trace('beta_2')[i]*pf_year_data + ResistanceSampler.trace('beta_3')[i]*suitability_data + ResistanceSampler.trace('beta_4')[i]*log10(1+population_data) + ResistanceSampler.trace('beta_5')[i]*log10(1+traveltime_data)
            else:
                Msurf_temp +=  ResistanceSampler.trace('beta_1')[i]*(j-2000) 
            Vsurf_temp += 1/ResistanceSampler.Vinv.value    
            freq = pm.invlogit(Msurf_temp + np.random.normal(0,1)*np.sqrt(Vsurf_temp))
            Mean_temp[where_unmasked2] = freq
            Msurf += map_utils.grid_convert(Mean_temp, 'y-x+','y+x+', validate = True)
            E_temp[where_unmasked2] = freq**2
            E2surf += map_utils.grid_convert(E_temp, 'y-x+','y+x+', validate = True)
            temp[:,:,count][where_unmasked2] = freq
            temp_cutoffs_30[count,j-first_year] = len(where(freq>0.3)[0])/len(freq)
            if save_rasters_all:
                close('all')
                figure()
                TempArray = ma.masked_array(map_utils.grid_convert(Mean_temp, 'y-x+','y+x+', validate = True), mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
                map_utils.export_raster(xplot,yplot,TempArray,'Resistance_plot_%s'%j+'_%s'%count,mypath,'flt',view='y-x+')
                imshow(TempArray, vmin=0., vmax=1., cmap = "jet",extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi,interpolation='nearest')
                savefig(my_fig_path+'Resistance_plot_%s'%j+'_%s'%count+'.pdf' )
            count += 1

        # ================================================================================
        # Get the posterior variance and standard deviation
        # ================================================================================
        Msurf = Msurf/float(count) 
        E2surf = E2surf/float(count) 
        Vsurf = E2surf - Msurf**2

        # ================================================================================
        # Covert back to the right grid orientation (y-x+)
        # ================================================================================
        MedSurf = map_utils.grid_convert(median(temp[:,:,range(count)],axis=2) , 'y-x+','y+x+', validate = True)
        IQR1Surf = map_utils.grid_convert(percentile(temp[:,:,range(count)], 25, axis=2) , 'y-x+','y+x+', validate = True)
        IQR2Surf = map_utils.grid_convert(percentile(temp[:,:,range(count)], 75, axis=2), 'y-x+','y+x+', validate = True) 

        # ================================================================================
        # Create masked array
        # ================================================================================
        Msurf = ma.masked_array(Msurf, mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
        MedSurf2 = ma.masked_array(MedSurf, mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
        IQR1Surf2 = ma.masked_array(IQR1Surf, mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
        IQR2Surf2 = ma.masked_array(IQR2Surf, mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))

        # ================================================================================
        # Create background surface
        # ================================================================================
        xplotRegion, yplotRegion, DataRegion, TypeRegion = map_utils.import_raster('AsiaRegion','/Users/jflegg/data/flt and hdr files for mapping codes', type = None) # these are in radians
        dataRegion = map_utils.grid_convert(ma.masked_array(DataRegion.data, mask = DataRegion.mask), 'y-x+','y+x+', validate = True)
        where_unmaskedRegion = np.where(1-dataRegion.mask.astype(np.float32)) 
        Bground = zeros(DataRegion.mask.shape) # is y+x+
        Bground[where_unmaskedRegion] = ones([len(where_unmaskedRegion[0])])*0.7  # is y+x+
        Bground = ma.masked_array(map_utils.grid_convert(Bground, 'y-x+','y+x+', validate = True), mask=DataRegion.mask) #is y-x+


        # ================================================================================
        # Plot median surfaces
        # ================================================================================
        close('all')
        figure()
        imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
        imshow(MedSurf2, vmin=0., vmax=1., cmap = "jet",extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi,interpolation='nearest')
        ind = (ResistanceSampler.year <= j)
        if len(where(ind==True)[0]) > 0:
            scatter(ResistanceSampler.lon[ind]*180./pi, ResistanceSampler.lat[ind]*180./pi, c=1.0*ResistanceSampler.number_with[ind]/ResistanceSampler.number_tested[ind], s=0.2*ResistanceSampler.number_tested[ind],  edgecolor='k',  cmap = "jet")     
        axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi)
        clim(0.0, 1.0)
        title('%s '%name_temp + 'posterior predictive median - %s'%j)
        xlabel('Longitude')
        ylabel('Latitude')
        colorbar()

        # plot the country borders....
        for nshp in range(Nshp):
            if (any(records[nshp][0]==ToPlot)):
                ptchs   = []
                pts     = np.array(shapes[nshp].points)
                prt     = shapes[nshp].parts
                par     = list(prt) + [pts.shape[0]]
                for pij in range(len(prt)):
                    plot(pts[par[pij]:par[pij+1]][:,0], pts[par[pij]:par[pij+1]][:,1], color = "black", linewidth = 0.5)

        savefig(my_fig_path+'Resistance_median_%s.pdf'%j)
        map_utils.export_raster(xplot,yplot,MedSurf2,'Resistance_median_%s'%j,mypath,'flt',view='y-x+')


        # ================================================================================
        # Plot threshold images 
        # ================================================================================
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thres in thresholds:
            close('all')
            figure()
            imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
            MedSurf2_temp = ones(MedSurf2.data.shape)*0.3
            ind = where(map_utils.grid_convert(MedSurf, 'y-x+','y+x+', validate = True) > thres)
            MedSurf2_temp[ind] = 0.9
            MedSurf2_temp = ma.masked_array(map_utils.grid_convert(MedSurf2_temp, 'y-x+','y+x+', validate = True), mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
            imshow(MedSurf2_temp, vmin=0., vmax=1., cmap = "Reds",extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi,interpolation='nearest')
            thres_temp = int(thres*100)
            savefig(my_fig_path+'Resistance_median_threshold_%s'%thres_temp+'_%s.pdf'%j)
            map_utils.export_raster(xplot,yplot,MedSurf2_temp,'Resistance_median_threshold_%s'%thres_temp+'_%s'%j,mypath,'flt',view='y-x+')


        # ================================================================================
        # Plot mean surfaces
        # ================================================================================
        close('all')
        figure()
        imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
        imshow(Msurf, vmin=0., vmax=1., cmap = "jet",extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi,interpolation='nearest')
        ind = (ResistanceSampler.year <= j)
        if len(where(ind==True)[0]) > 0:
            scatter(ResistanceSampler.lon[ind]*180./pi, ResistanceSampler.lat[ind]*180./pi, c=1.0*ResistanceSampler.number_with[ind]/ResistanceSampler.number_tested[ind], s=0.2*ResistanceSampler.number_tested[ind],  edgecolor='k',  cmap = "jet",)     


        axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi)
        clim(0.0, 1.0)
        title('%s '%name_temp + 'posterior predictive mean - %s'%j)
        xlabel('Longitude')
        ylabel('Latitude')
        colorbar()

        # plot the country borders....
        for nshp in range(Nshp):
            if (any(records[nshp][0]==ToPlot)):
                ptchs   = []
                pts     = np.array(shapes[nshp].points)
                prt     = shapes[nshp].parts
                par     = list(prt) + [pts.shape[0]]
                for pij in range(len(prt)):
                    plot(pts[par[pij]:par[pij+1]][:,0], pts[par[pij]:par[pij+1]][:,1], color = "black", linewidth = 0.5)


        savefig(my_fig_path+'Resistance_mean_%s.pdf'%j)
        map_utils.export_raster(xplot,yplot,MedSurf2,'Resistance_mean_%s'%j,mypath,'flt',view='y-x+')


        # ================================================================================
        # Try to get the standard deviation information for plotting, and then plot
        # ================================================================================
        try:
            # get standard deviation
            SDsurf = sqrt(Vsurf)
            # covert back to the right grid orientation 
            SDsurf2 = ma.masked_array(SDsurf,mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
            # ================================================================================
            # Plot standard deviation surfaces
            # ================================================================================
            close('all')
            figure()
            imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
            imshow(SDsurf2, vmin=0, vmax=0.5, cmap = "OrRd",extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi,interpolation='nearest')
            clim(0.0, 0.5)
            axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi)
            clim(SDsurf.min(), SDsurf.max())
            title('%s '%name_temp+ 'posterior predictive standard deviation - %s'%j)
            xlabel('Longitude')
            ylabel('Latitude')
            colorbar(cmap="OrRd")

            # plot the country borders....
            for nshp in range(Nshp):
                if (any(records[nshp][0]==ToPlot)):
                    ptchs   = []
                    pts     = np.array(shapes[nshp].points)
                    prt     = shapes[nshp].parts
                    par     = list(prt) + [pts.shape[0]]
                    for pij in range(len(prt)):
                        plot(pts[par[pij]:par[pij+1]][:,0], pts[par[pij]:par[pij+1]][:,1], color = "black", linewidth = 0.5)

            savefig(my_fig_path+'Resistance_sd_%s.pdf' %j)
            map_utils.export_raster(xplot,yplot,SDsurf2,'Resistance_sd_%s'%j,mypath,'flt',view='y-x+')
        except:
            print('could not calculate SDsurf')
            ####################### end plot ######################

        # ================================================================================
        # Plot IQR surfaces
        # ================================================================================
        matplotlib.pyplot.jet()
        subplot(1, 2, 1)
        imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
        imshow(IQR1Surf2, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi,interpolation='nearest')
        axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi)
        clim(0.0, 1.0)
        title('%s '%name_temp+'25th \n percentile - %s'%j)
        colorbar()
        # plot the country borders....
        for nshp in range(Nshp):
            if (any(records[nshp][0]==ToPlot)):
                ptchs   = []
                pts     = np.array(shapes[nshp].points)
                prt     = shapes[nshp].parts
                par     = list(prt) + [pts.shape[0]]
                for pij in range(len(prt)):
                    plot(pts[par[pij]:par[pij+1]][:,0], pts[par[pij]:par[pij+1]][:,1], color = "black", linewidth = 0.5)

        xlabel('Longitude')
        ylabel('Latitude')
        subplot(1, 2, 2)
        imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
        imshow(IQR2Surf2, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi,interpolation='nearest')
        axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180./pi)
        title('%s '%name_temp+'75th \n percentile - %s'%j)
        xlabel('Longitude')
        colorbar()
        # plot the country borders....
        for nshp in range(Nshp):
            if (any(records[nshp][0]==ToPlot)):
                ptchs   = []
                pts     = np.array(shapes[nshp].points)
                prt     = shapes[nshp].parts
                par     = list(prt) + [pts.shape[0]]
                for pij in range(len(prt)):
                    plot(pts[par[pij]:par[pij+1]][:,0], pts[par[pij]:par[pij+1]][:,1], color = "black", linewidth = 0.5)

        savefig(my_fig_path+'Resistance_iqr_%s_2.pdf' %j)
        map_utils.export_raster(xplot,yplot,IQR2Surf2,'Resistance_iqr_upper_%s'%j,mypath,'flt',view='y-x+')
        map_utils.export_raster(xplot,yplot,IQR1Surf2,'Resistance_iqr_lower_%s'%j,mypath,'flt',view='y-x+')
        ####################### end plot ######################

    # ================================================================================
    # Plot of actual versus predicted values at the data locations
    # ================================================================================
    years = ResistanceSampler.year
    if with_covariates:
        pf = ResistanceSampler.pf
        suitability = ResistanceSampler.suitability
        population = ResistanceSampler.population
        traveltime = ResistanceSampler.traveltime

    dplot2 = ResistanceSampler.data_mesh
    temp = zeros([dplot2.shape[0], int(math.ceil(float(n)/thin))])
    likelihoods = zeros([n])
    count = 0
    # Get E[v] and over the entire posterior
    for i in range(0,n, thin):
        # Reset all variables to their values at frame i of the trace
        ResistanceSampler.remember(0,i)
        Msurf_temp, Vsurf_temp = pm.gp.point_eval(ResistanceSampler.S.M_obs.value,ResistanceSampler.S.C_obs.value, dplot2)
        if with_covariates:
            Msurf_temp +=  ResistanceSampler.trace('beta_1')[i]*(years-2000) + ResistanceSampler.trace('beta_2')[i]*pf + ResistanceSampler.trace('beta_3')[i]*suitability + ResistanceSampler.trace('beta_4')[i]*log10(1+population) + ResistanceSampler.trace('beta_5')[i]*log10(1+traveltime)
        else:
            Msurf_temp +=  ResistanceSampler.trace('beta_1')[i]*(years-2000) 
        Vsurf_temp += 1.0/ResistanceSampler.Vinv.value    
        freq = pm.invlogit(Msurf_temp + np.random.normal(0,1)*np.sqrt(Vsurf_temp))
        temp[:,count] = freq
        likelihoods[count] = sum(pm.distributions.binomial_like(ResistanceSampler.number_with, ResistanceSampler.number_tested, freq))
        count += 1


    # ================================================================================
    # Get the posterior median 
    # ================================================================================
    Med = 100.*median(temp[:,range(count)],axis=1)
    R_1 = percentile(temp, 25, axis = 1)
    R_2 = percentile(temp, 75, axis = 1)


    # ================================================================================
    # Plot the actual versus precticted values
    # ================================================================================
    close('all')
    figure()
    resis = ResistanceSampler.number_with*(1./ResistanceSampler.number_tested)*100.
    
    plot(resis,Med,'k.',markersize=5)
    plot(range(100), range(100))
    axis(np.array([0,110,0,110]))
    title('Actual versus predicted')
    xlabel('Actual resistance (%)')
    ylabel('Predicted resistance (%)')
    savefig(my_fig_path+'Actual_vs_predicted_used_points.pdf')


    # ================================================================================
    # Print the correlation between predicted and actual values
    # ================================================================================
    try:
        correlation = np.corrcoef(Med, resis)[0,1]
        print(correlation)
    except:
        print('could not calculate correlation coefficient')
 

    try:
        # ================================================================================
        # Plot the likelihood over the traces
        # ================================================================================
        close('all')
        figure()
        plot(range(count),likelihoods[0:count],'k')
        title('Likelihood')
        xlabel('iteration')
        ylabel('Log ikelihood')
        savefig(my_fig_path+'Log ikelihood_final.pdf')
    except:
        print('could not plot likelihood values - FloatingPointError')

    meanY = mean(Med)
    Rsq = sum((Med-meanY)**2)/sum((resis - meanY)**2)
    print(Rsq)



    # ================================================================================
    # Plot the confidence intervals for the observations
    # ================================================================================
    m1= len(years)
    close('all')
    figure()
    outliers = []
    for i in range(m1):
        x = i
        ymin = R_1[i]
        ymax = R_2[i]
        axvline(x=x, ymin=ymin, ymax=ymax, color = 'r', linewidth = 1)
        dv = resis[i]/100
        if (dv < ymin) | (dv > ymax):
            outliers += [i]    

    resis = ResistanceSampler.number_with*(1./ResistanceSampler.number_tested)*100.
    plot(range(m1), resis ,'k.',markersize=5)
    plot(outliers, resis[outliers] ,'b.',markersize=5)
    title('')
    xlabel('Data point number')
    ylabel('CI and actual data point value')
    savefig(my_fig_path+'ConfInts_points.pdf')



    # ================================================================================
    # Plot the field_plus_nugget traces (only ones that exist!)
    # ================================================================================
    for i in range(len(ResistanceSampler.field_plus_nugget)):
        try: 
            close("all")
            figure(figsize=(10, 6))
            plot(range(len(ResistanceSampler.trace('field_plus_nugget%s'%i)[:])), ResistanceSampler.trace('field_plus_nugget%s'%i)[:])
            savefig(my_fig_path+"field_plus_nugget%s.pdf"%i)
        except Exception:
            pass


    # ================================================================================
    # Plot the parameter traces
    # ================================================================================
    if with_covariates:
        ToPlot = ['Vinv','beta_0','beta_1','beta_2', 'beta_3', 'beta_4','beta_5','tlc','log_scale','log_sigma','time_scale']
    else:
        ToPlot = ['Vinv','beta_0','beta_1','tlc','log_scale','log_sigma','time_scale']
    for PLOT in ToPlot:
        print(PLOT)
        close('all')
        figure()
        subplot(1, 2, 1)
        plot(range(len(ResistanceSampler.trace('%s'%PLOT)[:])), ResistanceSampler.trace('%s'%PLOT)[:])
        subplot(1, 2, 2)
        pm.Matplot.histogram(ResistanceSampler.trace('%s'%PLOT)[:], '%s'%PLOT, datarange=(None, None), rows=1, columns=2, num=2, last=True, fontmap={1:10, 2:8, 3:6, 4:5,5:4})
        savefig(my_fig_path+"%s_2.pdf"%PLOT)


    # ================================================================================
    # Plot the S_f_eval traces
    # ================================================================================
    m1 = ResistanceSampler.mesh_unique.shape[0]  # m1 here has be the number of UNIQUE space-time points!
    close('all')
    figure()
    for i in range(1,m1,2):
        plot(ResistanceSampler.trace('S_f_eval')[:,i])      

    xlabel('x')
    ylabel('S_f_eval')
    savefig(my_fig_path+'S_f_eval_2.pdf')


    # ================================================================================
    # Plot the autocorrlations
    # ================================================================================
    try:
        pm.Matplot.autocorrelation(ResistanceSampler, path=my_fig_path)
    except:
        print('could not calculate autocorrelation coefficients')


    # ================================================================================
    # Get trends over time for > 30% resistance
    # ================================================================================
    close('all')
    figure()
    plot(range(first_year,final_year), median(temp_cutoffs_30, axis=0),color='#CC4F1B')
    fill_between(range(first_year,final_year), percentile(temp_cutoffs_30, 25, axis=0), percentile(temp_cutoffs_30, 75, axis=0), alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
    savefig(my_fig_path+'trends.pdf')
    savetxt("trends_%s.csv"%name_temp, temp_cutoffs_30, delimiter=",")  




def plot_slices_validate(name, first_year, thin, i, mypath, data1, with_covariates, continent = "Africa"):
   
    # ================================================================================
    # Get a 'name' for saving the images
    # ================================================================================
    close('all')
    name_temp = os.path.split(name)[1][0:3]
    final_year = 2020


    # ================================================================================
    # Load the database and model
    # ================================================================================
    hf_name ='res_db_validation%s'%i+'.hdf5'
    prev_db = pm.database.hdf5.load(hf_name)
    ResistanceSampler = pm.MCMC(model.make_model(**data1),db=prev_db, dbmode='a', dbcomplevel=1,dbcomplib='zlib',dbname=hf_name)


    # ================================================================================
    # Countries borders (plotting)
    # ================================================================================
    sf = shapefile.Reader("/Users/jflegg/data/country borders/cn.dbf")
    records    = sf.records()
    shapes  = sf.shapes()
    Nshp    = len(shapes)
    ToPlot = array(["Thailand", "Myanmar","Cambodia","Viet Nam", "Lao People's Democratic Republic", "China", "Malaysia","Bangladesh", "Indonesia"])

    # ================================================================================
    # Get the lon, lat values across Africa
    # ================================================================================
    xplot, yplot, Data, Type = map_utils.import_raster('AsiaRegionMekong','/Users/jflegg/data/flt and hdr files for mapping codes', type = None) # these are in radians
    data_yp_xp = map_utils.grid_convert(ma.masked_array(Data.data, mask = Data.mask), 'y-x+','y+x+', validate = True)
    where_unmasked2 = np.where(1-data_yp_xp.mask.astype(np.float32))

 
    # ================================================================================
    # Set up the lon, lat, year to do the slice plotting 
    # ================================================================================
    dplot2 = dstack(meshgrid(xplot,yplot))[where_unmasked2]
    dplot2_temp = zeros([dplot2.shape[0],3])
    dplot2_temp[:,0]= dplot2[:,0]
    dplot2_temp[:,1]= dplot2[:,1]


    # ================================================================================
    # Get the suitability covariate, load suitability raster
    # ================================================================================
    if with_covariates:
        xplot_temp, yplot_temp, Data_suitability, Type = map_utils.import_raster('TemperatureSuitability','/Users/jflegg/data/temperature_suitability', type = None) 
        suitability_data = map_utils.grid_convert(Data_suitability.data, 'y-x+','y+x+', validate = True)[where_unmasked2]
        xplot_temp, yplot_temp, Data_population, Type = map_utils.import_raster('Population','/Users/jflegg/data/WorldPop', type = None) 
        population_data = map_utils.grid_convert(Data_population.data, 'y-x+','y+x+', validate = True)[where_unmasked2]
        xplot_temp, yplot_temp, Data_traveltime, Type = map_utils.import_raster('Accessibility2015','/Users/jflegg/data/2015_accessibility_to_cities', type = None) 
        Data_traveltime_temp = Data_traveltime.data # need an extra step to convert from integer to float
        Data_traveltime_temp = Data_traveltime_temp.astype(float)
        traveltime_data = map_utils.grid_convert(Data_traveltime_temp, 'y-x+','y+x+', validate = True)[where_unmasked2]


    # ================================================================================
    # How many saved traces there are:
    # ================================================================================
    n = len(ResistanceSampler.trace('Vinv')[:])


     # ================================================================================
    # Loop over each of the years to plot
    # ================================================================================
    for j in range(first_year,final_year):
        print('Plotting for t = %s'%j)
        times = j*np.ones(dplot2_temp.shape[0])
        dplot2_temp[:,2]=times


        # ================================================================================
        # Initialise the matrices for storing the output
        # ================================================================================
        Msurf = zeros(data_yp_xp.shape)
        Mean_temp = zeros(data_yp_xp.shape)
        E2surf = zeros(data_yp_xp.shape)
        E_temp = zeros(data_yp_xp.shape)
        temp = zeros([data_yp_xp.shape[0], data_yp_xp.shape[1],int(math.ceil(float(n)/thin))])


        # ================================================================================
        # Get the pf covariate for the right year 
        # ================================================================================
        if with_covariates:
            jtemp = j 
            if j < 2000:
                jtemp = 2000
            elif j > 2020:
                jtemp = 2020
            if continent == "Africa":
                xplot_temp, yplot_temp, Data_pf, Type = map_utils.import_raster('Africa_pfpr_%s'%jtemp, path_name_plot_K13 +'/data/202206_Global_Pf_Parasite_Rate_2000', type = None) 
            elif continent == "India":
                xplot_temp, yplot_temp, Data_pf, Type = map_utils.import_raster('India_pfpr_%s'%jtemp, path_name_plot_K13 +'/data/202206_Global_Pf_Parasite_Rate_2000', type = None) 
            elif continent == "Mekong":
                xplot_temp, yplot_temp, Data_pf, Type = map_utils.import_raster('Mekong_pfpr_%s'%jtemp, path_name_plot_K13 +'/data/202206_Global_Pf_Parasite_Rate_2000', type = None) 
    
            pf_year_data = map_utils.grid_convert(Data_pf.data, 'y-x+','y+x+', validate = True)[where_unmasked2]


        # ================================================================================
        # Get E[v] and E[v**2] over the entire posterior, working in y+x+ orientation
        # ================================================================================
        count = 0
        for i in range(0,n,thin):
            # Reset all variables to their values at frame i of the trace
            ResistanceSampler.remember(0,i)
            Msurf_temp, Vsurf_temp = pm.gp.point_eval(ResistanceSampler.S.M_obs.value,ResistanceSampler.S.C_obs.value, dplot2_temp)
            if with_covariates:
                Msurf_temp +=  ResistanceSampler.trace('beta_1')[i]*(j-2000) + ResistanceSampler.trace('beta_2')[i]*pf_year_data + ResistanceSampler.trace('beta_3')[i]*suitability_data + ResistanceSampler.trace('beta_4')[i]*log10(1+population_data) + ResistanceSampler.trace('beta_5')[i]*log10(1+traveltime_data)
            else:
                Msurf_temp +=  ResistanceSampler.trace('beta_1')[i]*(j-2000) 
            Vsurf_temp += 1/ResistanceSampler.Vinv.value    
            freq = pm.invlogit(Msurf_temp + np.random.normal(0,1)*np.sqrt(Vsurf_temp))
            Mean_temp[where_unmasked2] = freq
            Msurf += map_utils.grid_convert(Mean_temp, 'y-x+','y+x+', validate = True)
            E_temp[where_unmasked2] = freq**2
            E2surf += map_utils.grid_convert(E_temp, 'y-x+','y+x+', validate = True)
            temp[:,:,count][where_unmasked2] = freq
            count += 1

        # ================================================================================
        # Get the posterior variance and standard deviation
        # ================================================================================
        Msurf = Msurf/float(count) 
        E2surf = E2surf/float(count) 
        Vsurf = E2surf - Msurf**2

        # ================================================================================
        # Covert back to the right grid orientation (y-x+)
        # ================================================================================
        MedSurf = map_utils.grid_convert(median(temp[:,:,range(count)],axis=2) , 'y-x+','y+x+', validate = True)
        IQR1Surf = map_utils.grid_convert(percentile(temp[:,:,range(count)], 25, axis=2) , 'y-x+','y+x+', validate = True)
        IQR2Surf = map_utils.grid_convert(percentile(temp[:,:,range(count)], 75, axis=2), 'y-x+','y+x+', validate = True) 


        # ================================================================================
        # Create masked array
        # ================================================================================
        Msurf = ma.masked_array(Msurf, mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
        MedSurf2 = ma.masked_array(MedSurf, mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
        IQR1Surf2 = ma.masked_array(IQR1Surf, mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
        IQR2Surf2 = ma.masked_array(IQR2Surf, mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))


        # ================================================================================
        # Create background surface
        # ================================================================================
        xplotRegion, yplotRegion, DataRegion, TypeRegion = map_utils.import_raster('AsiaRegion','/Users/jflegg/data/flt and hdr files for mapping codes', type = None) # these are in radians
        dataRegion = map_utils.grid_convert(ma.masked_array(DataRegion.data, mask = DataRegion.mask), 'y-x+','y+x+', validate = True)
        where_unmaskedRegion = np.where(1-dataRegion.mask.astype(np.float32)) 
        Bground = zeros(DataRegion.mask.shape) # is y+x+
        Bground[where_unmaskedRegion] = ones([len(where_unmaskedRegion[0])])*0.7  # is y+x+
        Bground = ma.masked_array(map_utils.grid_convert(Bground, 'y-x+','y+x+', validate = True), mask=DataRegion.mask) #is y-x+


        # ================================================================================
        # Plot median surfaces
        # ================================================================================
        close('all')
        figure()
        imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
        imshow(MedSurf2, vmin=0., vmax=1., cmap = "jet",extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi,interpolation='nearest')
        ind = (ResistanceSampler.year <= j)
        if len(where(ind==True)[0]) > 0:
            scatter(ResistanceSampler.lon[ind]*180./pi, ResistanceSampler.lat[ind]*180./pi, c=1.0*ResistanceSampler.number_with[ind]/ResistanceSampler.number_tested[ind], s=0.2*ResistanceSampler.number_tested[ind],  edgecolor='k',  cmap = "jet",)     


        axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi)
        clim(0.0, 1.0)
        title('%s '%name_temp + 'posterior predictive median - %s'%j)
        xlabel('Longitude')
        ylabel('Latitude')
        colorbar()

        # plot the country borders....
        for nshp in range(Nshp):
            if (any(records[nshp][0]==ToPlot)):
                ptchs   = []
                pts     = np.array(shapes[nshp].points)
                prt     = shapes[nshp].parts
                par     = list(prt) + [pts.shape[0]]
                for pij in range(len(prt)):
                    plot(pts[par[pij]:par[pij+1]][:,0], pts[par[pij]:par[pij+1]][:,1], color = "black", linewidth = 0.5)


        savefig(mypath+'Resistance_median_%s.pdf'%j)
        map_utils.export_raster(xplot,yplot,MedSurf2,'Resistance_median_%s'%j,mypath,'flt',view='y-x+')


        # ================================================================================
        # Try to get the standard deviation information for plotting, and then plot
        # ================================================================================
        try:
            # get standard deviation
            SDsurf = sqrt(Vsurf)
            # covert back to the right grid orientation 
            SDsurf2 = ma.masked_array(SDsurf,mask=map_utils.grid_convert(data_yp_xp.mask , 'y-x+','y+x+', validate = True))
            # ================================================================================
            # Plot standard deviation surfaces
            # ================================================================================
            close('all')
            figure()
            imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
            imshow(SDsurf2, vmin=SDsurf.min(), vmax=SDsurf.max(), cmap = "OrRd",extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi,interpolation='nearest')

            axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi)
            clim(SDsurf.min(), SDsurf.max())
            title('%s '%name_temp+ 'posterior predictive standard deviation - %s'%j)
            xlabel('Longitude')
            ylabel('Latitude')
            colorbar(cmap="OrRd")

            # plot the country borders....
            for nshp in range(Nshp):
                if (any(records[nshp][0]==ToPlot)):
                    ptchs   = []
                    pts     = np.array(shapes[nshp].points)
                    prt     = shapes[nshp].parts
                    par     = list(prt) + [pts.shape[0]]
                    for pij in range(len(prt)):
                        plot(pts[par[pij]:par[pij+1]][:,0], pts[par[pij]:par[pij+1]][:,1], color = "black", linewidth = 0.5)

            savefig(mypath+'Resistance_sd_%s.pdf' %j)
            map_utils.export_raster(xplot,yplot,SDsurf2,'Resistance_sd_%s'%j,mypath,'flt',view='y-x+')
        except:
            print('could not calculate SDsurf')
            ####################### end plot ######################


def plot_from_raster(name, first_year, data, continent = "Africa"):

    # ================================================================================
    # Load the database and model
    # ================================================================================
    prev_db = pm.database.hdf5.load('res_db.hdf5')
    ResistanceSampler = pm.MCMC(model.make_model(**data),db=prev_db,dbmode='a', dbcomplevel=1,dbcomplib='zlib',dbname='res_db.hdf5')
    final_year = 2021

    # ================================================================================
    # Countries borders (plotting)
    # ================================================================================
    sf = shapefile.Reader("/Users/jflegg/data/country borders/cn.dbf")
    records    = sf.records()   
    shapes  = sf.shapes()
    Nshp    = len(shapes)
    ToPlot = array(["Thailand", "Myanmar","Cambodia","Viet Nam", "Lao People's Democratic Republic", "China", "Malaysia","Bangladesh", "Indonesia"])


    # ================================================================================
    #   Do plotting of slices from saved rasters 
    # ================================================================================
    mypath = 'rasters/'
    my_fig_path = 'figures/'
    name_temp = os.path.split(name)[1][0:3]

    # ================================================================================
    # Loop over each of the years to plot
    # ================================================================================
    for j in range(first_year,final_year):
        print('Plotting for t = %s'%j)

        # ================================================================================
        # Create background surface
        # ================================================================================
        xplotRegion, yplotRegion, DataRegion, TypeRegion = map_utils.import_raster('AsiaRegion','/Users/jflegg/data/flt and hdr files for mapping codes', type = None) # these are in radians
        dataRegion = map_utils.grid_convert(ma.masked_array(DataRegion.data, mask = DataRegion.mask), 'y-x+','y+x+', validate = True)
        where_unmaskedRegion = np.where(1-dataRegion.mask.astype(np.float32)) 
        Bground = zeros(DataRegion.mask.shape) # is y+x+
        Bground[where_unmaskedRegion] = ones([len(where_unmaskedRegion[0])])*0.7  # is y+x+
        Bground = ma.masked_array(map_utils.grid_convert(Bground, 'y-x+','y+x+', validate = True), mask=DataRegion.mask) #is y-x+


        # ================================================================================
        # Plot median surfaces
        # ================================================================================
        xplot, yplot, MedSurf2, Type = map_utils.import_raster('Resistance_median_%s'%j,mypath, type = None) # these are in radians
        close('all')
        figure()
        imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
        imshow(MedSurf2, vmin=0., vmax=1., cmap = "jet",extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi,interpolation='nearest')
        ind = (ResistanceSampler.year <= j)
        if len(where(ind==True)[0]) > 0:
            scatter(ResistanceSampler.lon[ind]*180./pi, ResistanceSampler.lat[ind]*180./pi, c=1.0*ResistanceSampler.number_with[ind]/ResistanceSampler.number_tested[ind], s=0.2*ResistanceSampler.number_tested[ind],  edgecolor='k',  cmap = "jet",)     

        axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi)
        clim(0.0, 1.0)
        title('%s '%name_temp + 'posterior predictive median - %s'%j)
        xlabel('Longitude')
        ylabel('Latitude')
        colorbar()

        # plot the country borders....
        for nshp in range(Nshp):
            if (any(records[nshp][0]==ToPlot)):
                ptchs   = []
                pts     = np.array(shapes[nshp].points)
                prt     = shapes[nshp].parts
                par     = list(prt) + [pts.shape[0]]
                for pij in range(len(prt)):
                    plot(pts[par[pij]:par[pij+1]][:,0], pts[par[pij]:par[pij+1]][:,1], color = "black", linewidth = 0.5)

        savefig(my_fig_path+'Resistance_median_%s.pdf'%j)


        # ================================================================================
        # Try to get the standard deviation information for plotting, and then plot
        # ================================================================================
        try:
            # ================================================================================
            # Plot standard deviation surfaces
            # ================================================================================
            close('all')
            figure()
            xplot, yplot, SDsurf2, Type = map_utils.import_raster('Resistance_sd_%s'%j,mypath, type = None) # these are in radians
            imshow(Bground, vmin=0., vmax=1., extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi, cmap="gist_gray",interpolation='nearest') # visualise y-x+
            imshow(SDsurf2, vmin=0, vmax=0.35, cmap = "OrRd",extent=np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi,interpolation='nearest')

            axis(np.array([xplot.min(),xplot.max(),yplot.min(),yplot.max()])*180.0/pi)
            clim(SDsurf2.min(), SDsurf2.max())
            title('%s '%name_temp+ 'posterior predictive standard deviation - %s'%j)
            xlabel('Longitude')
            ylabel('Latitude')
            colorbar(cmap="OrRd")


            # plot the country borders....
            for nshp in range(Nshp):
                if (any(records[nshp][0]==ToPlot)):
                    ptchs   = []
                    pts     = np.array(shapes[nshp].points)
                    prt     = shapes[nshp].parts
                    par     = list(prt) + [pts.shape[0]]
                    for pij in range(len(prt)):
                        plot(pts[par[pij]:par[pij+1]][:,0], pts[par[pij]:par[pij+1]][:,1], color = "black", linewidth = 0.5)

            savefig(my_fig_path+'Resistance_sd_%s.pdf' %j)
        except:
            print('could not calculate SDsurf')
        ####################### end plot ######################






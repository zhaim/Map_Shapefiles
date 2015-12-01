
# coding: utf-8

# # Preparation

# In[1]:

import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import shapely.geometry as geom
import ast as ast
import datetime as dt
from PIL import Image as im
import shapefile
from matplotlib.collections import PatchCollection, PolyCollection
from mpl_toolkits.basemap import Basemap
from shapely.geometry import Point, MultiPoint, MultiPolygon
from descartes import PolygonPatch
from pysal.esda.mapclassify import Natural_Breaks as nb
mpl.style.use('ggplot')


data = pd.read_csv("data_sci_snippet.csv")
zips = pd.read_csv("zips.csv")
zips['Poly']=zips['geojson'].map(lambda x: geom.Polygon(ast.literal_eval(x)['coordinates'][0][0]))
zips['Verts']=zips['geojson'].map(lambda x: ast.literal_eval(x)['coordinates'][0][0])
data['Coor'] = data.apply(lambda x: geom.Point((x['GeoLon']),(x['GeoLat'])), axis = 1)

# Join the two df without common key column
data['key'] = 1
zips['key'] = 1
joined = pd.merge(data, zips, on='key').ix[:,:]
joined['Within'] = joined.apply(lambda x: x["Poly"].contains(x["Coor"]), axis = 1)
joined = joined[joined.Within == True]
joined.reset_index(level=0,inplace=True)

# Add new columns
joined['ListD'] = joined.apply(lambda x: dt.datetime.strptime(x.ListDate,"%Y-%m-%d"), axis =1) 
joined['ListY'] = joined.apply(lambda x: x['ListD'].year, axis =1)
joined['ListM'] = joined.apply(lambda x:  str(x['ListY']) + '-' + str(x['ListD'].strftime('%m')), axis =1)

print 'If the CloseDate is NaN, update it to ListDate (DoM will be 0).\n'
joined['CloseDate'].fillna(joined['ListDate'], inplace=True)
joined['CloseD'] = joined.apply(lambda x: dt.datetime.strptime(x.CloseDate,"%Y-%m-%d"), axis =1)
joined['CloseY'] = joined.apply(lambda x: x['CloseD'].year, axis =1)
joined['CloseM'] = joined.apply(lambda x:  str(x['CloseY']) + '-' + str(x['CloseD'].strftime('%m')), axis =1)

# Convenience functions to draw bar chart and map
def draw_bar_chart(df,x_axis,x_label, y_axis,y_label,Color,Title):
    plt.clf()
    fig = plt.figure()
    df.plot(kind="Bar", x= x_axis,y = y_axis, color = Color)
    plt.title(Title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45)
    fig.set_size_inches(10,10)
    plt.savefig(Title, ext='png', close=True, dpi=400, bbox_inches='tight')
    
def draw_map(df, measure, Colors,Title):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111,axisbg='None')
    ax.set_xlim(minx - 0.2 * w, maxx + 0.2 * w)
    ax.set_ylim(miny - 0.2 * h, maxy + 0.2 * h)
    ax.set_aspect(1)

    cmap = plt.get_cmap(Colors)
    df['patches'] = df['Poly'].map(lambda x: PolygonPatch(x, ec='#555555', lw=0.2, alpha=0.5, zorder=4))
    pc = PatchCollection(df['patches'].values, match_original = True)
    norm = mpl.colors.Normalize()
    pc.set_facecolor(cmap(norm(df[measure].values)))
    ax.add_collection(pc)
    plt.title(Title)

    #Add a colorbar for the PolyCollection
    #Classified the prices into 7 classes using natural break
    breaks = nb(
        df[df[measure].notnull()][measure].values,
        initial=500,
        k=7)
    jb = pd.DataFrame({'jenks_bins': breaks.yb}, index=df[df[measure].notnull()].index)
    df = df.join(jb)
    df.jenks_bins.fillna(-1, inplace=True)
    jenks_labels = ["<= Above %0.1f" % b for b in breaks.bins]
    cb = colorbar_index(ncolors=len(jenks_labels), cmap=cmap, shrink=0.5, labels=jenks_labels)
    cb.ax.tick_params(labelsize=8)

    fig.set_size_inches(10,10)
    plt.savefig(Title, ext='png', close=True, dpi=400, bbox_inches='tight')   
    
    

# Convenience functions for working with colour ramps and bars
def colorbar_index(ncolors, cmap, labels=None, **kwargs):
    """
    This is a convenience function to stop you making off-by-one errors
    Takes a standard colour ramp, and discretizes it,
    then draws a colour bar with correctly aligned labels
    """
    cmap = cmap_discretize(cmap, ncolors)
    mappable = mpl.cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, **kwargs)
    colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
    colorbar.set_ticklabels(range(ncolors))
    if labels:
        colorbar.set_ticklabels(labels)
    return colorbar

def cmap_discretize(cmap, N):
    """
    Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)

    """
    if type(cmap) == str:
        cmap = get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0., 0., 0., 0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N + 1)
    cdict = {}
    for ki, key in enumerate(('red', 'green', 'blue')):
        cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki]) for i in xrange(N + 1)]
    return mpl.colors.LinearSegmentedColormap(cmap.name + "_%d" % N, cdict, 1024)


mPriceByPoly = pd.DataFrame.merge(mPriceByZip, zips, how='left', on=['postal_code'], sort=True)


# Create a shapefile
wt = shapefile.Writer(shapefile.POLYGON)
wt.poly(parts=mPriceByPoly['Verts'].tolist())
wt.field('FIRST_FLD','C','40',0)
wt.record('First','Polygon')
wt.save("shapefile_phnx")

# Now I can create a basemap instance which I could draw map on

# lower left minx miny , upper right maxx maxy
bounds = [min(joined['GeoLon']),min(joined['GeoLat']), 
          max(joined['GeoLon']),max(joined['GeoLat'])]
minx, miny, maxx, maxy = bounds
w, h = maxx - minx, maxy - miny

# add a Basemap instance and a small additional extent to the boundry
m = Basemap(
    projection='tmerc',
    ellps = 'WGS84',
    llcrnrlon=minx - 0.2 * w,
    llcrnrlat=miny - 0.2 * h,
    urcrnrlon=maxx + 0.2 * w,
    urcrnrlat=maxy + 0.2 * h,
    lat_0=miny+0.5*h,
    lon_0=minx+0.5*w,
    lat_ts = 0,
    resolution='h',
    suppress_ticks=True)

#Add the shapefile Made earlier to the basemap
m.readshapefile('shapefile_phnx','phnx',zorder=2) 


print '\n1.1.3 - Medium list price by zip codes graphically\n'
mPriceByPoly = pd.DataFrame.merge(mPriceByZip, zips, how='left', on=['postal_code'], sort=True)
draw_map(mPriceByPoly,'ListPrice','Reds','Median List Price by Zip Code Graphically')


print """\n    From this map I could say that:
    1. Northeast has higher prices than elsewhere
    2. North has higher prices than south
    3. There is an area with exremely high price in the middle(Maybe near downtown)"""

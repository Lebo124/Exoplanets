
# coding: utf-8

# # Final Assignement
# 
# For the final assignement i was curious how unsupervised learning could help in clustering data. I was impressed of all the beautiful things that could be made by the neural networks, like deep dreaming, but i also am curious about the way deep learning can be used in science.
# Another hobby of mine is astronomy and in particular i am interested in the discovery of exoplanets (planets around other stars) the last 20 years.
# The exoplanets are divided in groups based on the planets oif our own solarsystem (Mercurian, Subterran, Terran, SuperTerran, Neptunian and Jovian). Off course that is what we people tend to do when we discover new worlds, fit thme in the old ones.
# Maybe when a computer does the dividing in categories it comes up with a different kind of structure that is based on the data.
# This is my effort to try something like that. 
# First i will import different kind of things. Some of them like tensorflow and numpy i imported first and while going further in my investigation i also discovered new libraries in Python i did not know like collections and operator.

# In[1]:


import tensorflow as tf
import numpy as np
import pandas as pd
import gif
from sklearn import preprocessing
from matplotlib import pyplot as plt
from collections import Counter
from operator import itemgetter


get_ipython().magic('matplotlib inline')


# I used a dataset which contains a Mass Class variable that has allready divided the mass of the planets in groups 
# compared with the planets in our own solarsystem (Jovian, Neptunian, Superterran, Terran, SubTerran an Mercurian. So i can compare the classification that the network makes (if at all possible) with this classification. I will put the database in the zip file. But you can download it yourself, because almost every day there are more exoplanets discovered.
# download from http://phl.upr.edu/projects/habitable-exoplanets-catalog/data/database
# The database exists out of 65 variables. I only needed a couple of them. So i deleted the ones i did not need and made a new database out of the variables i needed.
# What variables i did use?
# Only those who are obtained directly by observation. Not the ones that are derives from them.
# 
# # If you doubleclick on this field you will see the names as a table!!
# 
# This are the variables of the database.
#  #	 Field Name	            Description	Units
#  01	 P. Name	            Planet name.	 
#  02	 P. Name Kepler	        Planet NASA Kepler name, if applicable.	 
#  03	 P. Name KOI	        Planet NASA Kepler Object of Interest (KOI) name, if applicable.	 
#  04	 P. Zone Class	        Planet habitable zone classification (hot, warm, or cold).	 
#  05	 P. Mass Class	        Planet Mass Class (mercurian, subterran, terran, superterran, neptunian, or jovian)
#  06	 P. Composition Class	Planet Composition Class (iron, rocky-iron, rocky-water, water-gas, gas).	 
#  07	 P. Atmosphere Class	Planet Atmosphere Class (none, metals-rich, hydrogen-rich).	 
#  08	 P. Habitable Class	    Planet Habitable Class (mesoplanet, thermoplanet, psychroplanet, hypopsychroplanet,                                 hyperthermoplanet, non-habitable).	 
#  09	 P. Min Mass	        Planet minimum mass.	Earth Units, EU
#  10	 P. Mass	            Planet mass. Most of the values were estimated from minimum mass.	Earth Units, EU
#  11	 P. Max Mass	        Planet maximum mass.	Earth Units, EU
#  12	 P. Radius	            Planet radius. Most of these values were estimated for confirmed planets.	Earth                                    Units, EU
#  13	 P. Density	            Planet density.	Earth Units, EU
#  14	 P. Gravity	            Planet gravity.	Earth Units, EU
#  15	 P. Esc Vel	            Planet escape velocity.	Earth Units, EU
#  16	 P. SFlux Min	        Planet minimum stellar flux.	Earth Units, EU
#  17	 P. SFlux Mean	        Planet mean stellar flux.	Earth Units, EU
#  18	 P. SFlux Max	        Planet maximum stellar flux.	Earth Units, EU
#  19	 P. Teq Min	            Planet minimum equilibrium temperature (at apastron).	Kelvins, K
#  20	 P. Teq Mean	        Planet mean equilibrium temperature.	Kelvins, K
#  21	 P. Teq Max	            Planet maximum equilibrium temperature (at periastron).	Kelvins, K
#  22	 P. Ts Min	            Planet minimum surface temperature (at apastron).	Kelvins, K
#  23	 P. Ts Mean	            Planet mean surface temperature.	Kelvins, K
#  24	 P. Ts Max	            Planet maximum surface temperature (at periastron).	Kelvins, K
#  25	 P. Surf Press	        Planet surface pressure.	Earth Units, EU
#  26	 P. Mag	Planet          magnitude as seen from a Moon-Earth distance (Moon = -12.7).	
#  27	 P. Appar Size	        Planet apparent size as seen from a Moon-Earth distance (Moon = 0.5°).	degrees
#  28	 P. Period	Planet      period.	days
#  29	 P. Sem Major Axis	    Planet semi major axis.	Astr. Units, AU
#  30	 P. Eccentricity	    Planet eccentricity (assumed 0 when not available).	
#  31	 P. Mean Distance       Planet mean distance from the star.	Astr. Units, AU
#  32	 P. Inclination	        Planet inclination (assumed 60° when not available, and 90° for Kepler data).	degrees
#  33	 P. Omega	            Planet omega.	degrees
#  34	 S. Name	            Star name.	
#  35	 S. Name HD	            Star name from the Henry Draper Star Catalog.	 
#  36	 S. Name HIP	        Star name from the Hipparchus Star Catalog.	 
#  37	 S. Constellation	    Star constellation name (abbreviated).	 
#  38	 S. Type	            Star type.	
#  39	 S. Mass	            Star mass.	Solar Units, SU
#  40	 S. Radius	            Star radius.	Solar Units, SU
#  41	 S. Teff	            Star effective temperature.	Solar Units, SU
#  42	 S. Luminosity	        Star luminosity.	Solar Units, SU
#  43	 S. [Fe/H]	            Star iron to hydrogen ratio [Fe/H].	
#  44	 S. Age	Star age.	    Billion Years, Gyrs
#  45	 S. Appar Mag	        Star apparent visual magnitude from Earth.	
#  46	 S. Distance	        Star distance from Earth.	parsec, pc
#  47	 S. RA	Star            right ascension.	degrees
#  48	 S. DEC	Star            declination.	    degrees
#  49	 S. Mag from Planet	    Star apparent visual magnitude from planet (Sun = -25 from Earth).	
#  50	 S. Size from Planet	Star apparent size from planet (Sun = 0.5° from Earth).	degrees
#  51	 S. No. Planets	        Star number of planets.	 
#  52	 S. No. Planets	        Star number of planets in the habitable zone.	 
#  53	 S. Hab Zone Min	    Star inner edge of habitable zone.	Astr. Units, AU
#  54	 S. Hab Zone Max	    Star outer edge of habitable zone.	Astr. Units, AU
#  55	 P. HZD	Planet          Habitable Zone Distance (HZD).	
#  56	 P. HZC	Planet          Habitable Zone Composition (HZD).	
#  57	 P. HZA	Planet          Habitable Zone Atmosphere (HZD).	
#  58	 P. HZI	Planet          Habitable Zone Index (HZI).	 
#  59	 P. SPH	                Planet Standard Primary Habitability (SPH).	
#  60	 P. Int ESI	            Planet Interior Earth Similarity Index (iESI).	
#  61	 P. Surf ESI	        Planet Surface Earth Similarity Index (sESI).	
#  62	 P. ESI	                Planet Earth Similarity Index (ESI).	
#  63	 S. HabCat	            Star is in the HabCat database.	FLAG = 0 or 1
#  64	 P. Habitable	        Planet is potentially habitable.	FLAG = 0 or 1 
#  65	 P. Hab Moon	        Planet is candidate for potential habitable exomoons.	FLAG = 0 or 1
#  66	 P. Confirmed	        Planet is confirmed (relevant to the Kepler data).	FLAG = 0 or 1
#  67	 P. Disc. Method	    Planet discovery method (rv = radial velocity, tran = transiting, ima = imaging, micro =                           micro lensing, pul = pulsar timing, ast = astrometry).	 
#  68	 P. Disc. Year	        Planet discovery year.	 
# 
# I used the following variables and changed there names because in the origin names blanks and () are used and that leads to trouble in Python.
# 
# PmassEU            Planet Mass in comparison with Earth Mass (=1.0)
# PradiusEU          Planet Radius in comparison with radius of the Earth(1.0)
# PdensityEU         Density of the planet in comparison with density of the Earth(1.0)
# PgravityEU         Gravity of the planet in comparison with gravity of the Eartg(1.0)
# PfluxMeanEU        Flux of the planet is comparison to the Earth (1.0)	
# PteqMeanK	       Mean equilibrium of the planet in degrees Kelvin
# PsurfPressEU       Pressure on the surface in comparison to the Earth(1.0)
# Pmag	           Magnitude from the planet as seen from distance moon Earth (0.5 degeers is magnitude moon seen                      from Earth)
# PperiodDYS         Orbital period from planet in days
# SmassSU            Mass hoststar compared to the sun (1.0)
# SradiusSU          Radius from hoststar in comparison with the sun (1.0)	
# SteffK             Temperature of the hoststar in degrees Kelvin
# SluminositySU      Luminosity of the hoststar compared to the sun (1.0)
# SsizefromPlanetDEG Size from the hoststar seen from the planet in  degrees.
# 
# 
# 
# 
# 
# 
# 

# In[2]:


#read the database (take care the database is in the same map as the notebook)
data = pd.read_csv('Exoplanetsconf.csv', low_memory=False)

# the number of rows and of columns
data=data.dropna()
print (len(data)) #number of observations (rows)
print (len(data.columns)) # number of variables (columns)"


# In[3]:


# Normalize data between 0-1

x = data.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
datanorm = x_scaled


# # Neural Network
# 
# It took some time and some searching on the internet to decide which network can best be used. At last i stumbled upon the "selforganised maps" also known as the Kohonen maps. https://en.wikipedia.org/wiki/Self-organizing_map
# and http://ieeexplore.ieee.org/document/58325/
# This kind of networks can be used to find structures in data that is not labeled. 
# The idea of this network is that it learns by 'competitive learning'.
# Because there is no labeling there is no cost function. What happens is that the output layer has the number of neurons equal to the number of groups you want to divide the data in (off course you often dont know how many groups is best, but you can try different number of groups and see what number of groups is most meaningfull)
# You can put the output neurons in a 2D layer, so later you can make a map of the data in 2D.
# At first all the weights get random numbers, the first inputvector is put in the network and one of the output neurons scores the best. This neuron gets the inputvector assigned to it. The weights are changed so that that vector has more change to get an inputvector assigned that is like the first one. The other weights are changed so that all the other neurons have less chance to get the same kind of vector assigned. So  all inputvectors  go to through the network several times. The data is more and more centered around the neurons (centroïds).
# The input and outputvectors are fully connected and there are no hidden layers. A very simple and understandable network so.
# On this adress i found https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
# code for a SOM (selforganising network). I had to change it for my own use, but i used most of the code you see below from that pages. Thanks to Sachlin Joglekar !
# With the comments it is not hard to understand.
# I added the last three methods myself to make it possible to plot figures and do a reseach on the different groups.
# 

# In[5]:


imgs=[]
 
class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function
    and linearly decreasing learning rate.
    """
 
    #To check if the SOM has been trained
    _trained = False
 
    def __init__(self, m, n, dim, n_iterations=100, alpha=None, sigma=None):
        """
        Initializes all necessary components of the TensorFlow
        Graph.
 
        m X n are the dimensions of the SOM. 'n_iterations' should
        should be an integer denoting the number of iterations undergone
        while training.
        'dim' is the dimensionality of the training inputs.
        'alpha' is a number denoting the initial time(iteration no)-based
        learning rate. Default value is 0.3
        'sigma' is the the initial neighbourhood value, denoting
        the radius of influence of the BMU while training. By default, its
        taken to be half of max(m, n).
        """
 
        #Assign required variables first
        self._m = m
        self._n = n
        imgs=[]
        if alpha is None:
            alpha = 0.3
        else:
            alpha = float(alpha)
        if sigma is None:
            sigma = max(m, n) / 2.0
        else:
            sigma = float(sigma)
        self._n_iterations = abs(int(n_iterations))
 
        ##INITIALIZE GRAPH
        self._graph = tf.Graph()
 
        ##POPULATE GRAPH WITH NECESSARY COMPONENTS
        with self._graph.as_default():
 
            ##VARIABLES AND CONSTANT OPS FOR DATA STORAGE
 
            #Randomly initialized weightage vectors for all neurons,
            #stored together as a matrix Variable of size [m*n, dim]
        #HERE i used random_uniform instead of random_normal in the original code.(LB)
            self._weightage_vects = tf.Variable(tf.random_uniform(
                [m*n, dim]))
            
            #Matrix of size [m*n, 2] for SOM grid locations
            #of neurons
            self._location_vects = tf.constant(np.array(
                list(self._neuron_locations(m, n))))
 
            ##PLACEHOLDERS FOR TRAINING INPUTS
            #We need to assign them as attributes to self, since they
            #will be fed in during training
 
            #The training vector
            self._vect_input = tf.placeholder("float", [dim])
            #Iteration number
            self._iter_input = tf.placeholder("float")
 
            ##CONSTRUCT TRAINING OP PIECE BY PIECE
            #Only the final, 'root' training op needs to be assigned as
            #an attribute to self, since all the rest will be executed
            #automatically during training
 
            #To compute the Best Matching Unit given a vector
            #Basically calculates the Euclidean distance between every
            #neuron's weightage vector and the input, and returns the
            #index of the neuron which gives the least value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weightage_vects, tf.stack(
                    [self._vect_input for i in range(m*n)])), 2), 1)),
                                  0)
 
            #This will extract the location of the BMU based on the BMU's
            #index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vects, slice_input,
                                          tf.constant(np.array([1, 2]))),
                                 [2])
 
            #To compute the alpha and sigma values based on iteration
            #number
            learning_rate_op = tf.subtract(1.0, tf.divide(self._iter_input,
                                                  self._n_iterations))
            _alpha_op = tf.multiply(alpha, learning_rate_op)
            _sigma_op = tf.multiply(sigma, learning_rate_op)
 
            #Construct the op that will generate a vector with learning
            #rates for all neurons, based on iteration number and location
            #wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vects, tf.stack(
                    [bmu_loc for i in range(m*n)])), 2), 1)
            neighbourhood_func = tf.exp(tf.neg(tf.divide(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)
 
            #Finally, the op that will use learning_rate_op to update
            #the weightage vectors of all neurons based on a particular
            #input
            learning_rate_multiplier = tf.stack([tf.tile(tf.slice(
                learning_rate_op, np.array([i]), np.array([1])), [dim])
                                               for i in range(m*n)])
            weightage_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._vect_input for i in range(m*n)]),
                       self._weightage_vects))                                         
            new_weightages_op = tf.add(self._weightage_vects,
                                       weightage_delta)
            self._training_op = tf.assign(self._weightage_vects,
                                          new_weightages_op)                                       
 
            ##INITIALIZE SESSION
            self._sess = tf.Session()
 
            ##INITIALIZE VARIABLES
            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
 
    def _neuron_locations(self, m, n):
        """
        Yields one by one the 2-D locations of the individual neurons
        in the SOM.
        """
        #Nested iterations over both dimensions
        #to generate all 2-D locations in the map
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])
 
    def train(self, input_vects):
        """
        Trains the SOM.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Current weightage vectors for all neurons(initially random) are
        taken as starting conditions for training.
        """
 
        #Training iterations
        for iter_no in range(self._n_iterations):
            print(iter_no)
            if (iter_no % 1==0) & (iter_no>0) :
                
                self.map_plot(iter_no)
            centroid_grid = [[] for i in range(self._m)]
            self._weightages = list(self._sess.run(self._weightage_vects))
            self._locations = list(self._sess.run(self._location_vects))
            
            for i, loc in enumerate(self._locations):
                centroid_grid[loc[0]].append(self._weightages[i])
            self._centroid_grid = centroid_grid        
            
            #Train with each vector one by one
            for input_vect in input_vects:
                self._sess.run(self._training_op,
                               feed_dict={self._vect_input: input_vect,
                                          self._iter_input: iter_no})
        print(iter_no)
        self.map_plot(iter_no)                              
        self._trained = True
        gif.build_gif(imgs, saveto='exoplaneta005s6 .gif')
        
        
    def get_centroids(self):
        """
        Returns a list of 'm' lists, with each inner list containing
        the 'n' corresponding centroid locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        return self._centroid_grid
 
    def map_vects(self, input_vects):
        """
        Maps each input vector to the relevant neuron in the SOM
        grid.
        'input_vects' should be an iterable of 1-D NumPy arrays with
        dimensionality as provided during initialization of this SOM.
        Returns a list of 1-D NumPy arrays containing (row, column)
        info for each input vector(in the same order), corresponding
        to mapped neuron.
        """

        if not self._trained:
            raise ValueError("SOM not trained yet")
 
        to_return = []
        for vect in input_vects:
            min_index = min([i for i in range(len(self._weightages))],
                            key=lambda x: np.linalg.norm(vect-
                                                         self._weightages[x]))
            to_return.append(self._locations[min_index])
 
        return to_return
# I  asserted this code to make the SOM draw a map of the groups every 10 iterations.
# the size of the red circles are the size of the different groups.
    def map_plot(self, iter_no):
        """
        Makes a plot of the groups found. Plots groups as a circle. Radius circle is size group.   
        """
        
        m = self._m
        n = self._n
        plt.figure()
        label=np.zeros(m*n)
        self._trained = True
        mapped = self.map_vects(datanorm)
        mapped=tuple(map(tuple, mapped))
        c=Counter(mapped)
        
        c= sorted(c.items(), key=itemgetter(1))
        a=[m*n]
        for i in range(0,len(c)):
            x=(((c[i])[0])[0])
            y=(((c[i])[0])[1])
            z=((c[i])[1])
            plt.plot(x, y, 'ro', markersize= z/(2*m*n))  
        plt.savefig('exoplanet{}.png'.format(iter_no))
        p=plt.imread('exoplanet{}.png'.format(iter_no))
        imgs.append(p)
        plt.show()
        plt.close()
        print(c)
        self._trained = False
        
    
    def group_matrix(self, group):
        """
        Makes a plot.scatter of group in (). Shows all variables plot to another.  
        """
        mapped = self.map_vects(datanorm)
        mappednp= np.array(mapped)
        groups= mappednp[:,0]
        data['Group'] = pd.Series(groups, index=data.index)
        pd.scatter_matrix(data[data['Group']==group], alpha = 0.3, figsize = (28,28));
        
    def group_describe(self, group):
        """
        Describes group in (). Uses pd.describe(). Gives Mean, std, min, max and 25,75,75%   
        """
        mapped = self.map_vects(datanorm)
        mappednp= np.array(mapped)
        
        groups= mappednp[:,0]
        data['Group'] = pd.Series(groups, index=data.index)
        print(data[data['Group']==group].describe())
        


# Here i train the network in a grid of 12 x 1 (so 12 possible groups) 13 is the dimension of the inputvector 
# (13 variables) and the last number is the number of iterations. I also experimented with the alpha and the 
# sigma, because with the Every iteration a new map is drawn and at 
# the end a gif called 'exogif.gif' is made 

# In[ ]:


som = SOM(12, 1 , 13, 1600,alpha=0.05, sigma=4)
#print(som._location_vects)
som.train(datanorm)


# I let the network run several times with different configuaration ( different alpha and sigma values, also differend numbers of groups). In the gif you see thet the network first seems unstable, groups jump from left to right and so on. This has to do with the alpha setting. You can look at the alpha setting as a kind of learning rate. When it is to high the network gets unstable not only in the beginning, but also later on several times. Even  when i tried more than 2000 iterations the groups did not get stable. If the sigma is too high the grouop that is the biggest in the beginning is drawing too much data too it, like a black hole. You can see the sigma number as a kjind of attraction force of the groups.
# With this configuration i get 12 groups, two of them are smaller than 100, the rest is between 100 and 732. 
# 
# Now we have some groups, but the question is are they meaningfull and what do these groups represent. To research that i will extend the inputvectors with a label of the different groups.
# I made two methods to examine the groups som.groups_describe(group) and som.group_matrix(group).
# You can describe and make a Matrix of one group, below that you can see the matrix of the complete dataset.
# 
# 
# I am not a scientist so i cannot say that this groups have scientific value, but you will see that in different groups there are in the matrix nice functions visible that are not visible in the complete dataset.
# 
# There is a lot of work to do to know if it is usefull to use neural networks to examine this kind of data. Also it is possible to use other neural networks. But in the framework of this course it will move to far to go into this.
# 

# In[ ]:


som.group_describe(4)
som.group_matrix(4)



# In[ ]:


# this matrix if of the complete dataset.
pd.scatter_matrix(data, alpha = 0.3, figsize = (28,28));


# I liked the course very much. Thanks !!!!
# 

# In[ ]:





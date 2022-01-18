from numpy.core.fromnumeric import compress, shape
from numpy.lib.shape_base import expand_dims
#te trau la shockwave en els steps
steps = np.arange(6500, 12720, 10)

steps = np.arange(4000, 6500, 10)

theta = 32 # pla equatorial
minshock = np.empty([0, 2])
shocksrad = np.empty([128,2,0])
timeshock = np.empty([0,1])
epsilon = 0.15#l'original està fet amb epsilon = 0.15
r = 10**3.8 #radi a partir del cual comença a baixar

for t in steps:
    model = rdAe(########################################)
    if t == steps[0]:
        x_grid = np.array([np.array(model['grid']['x']['znl'])[4:-4],np.array(model['grid']['x']['znc'])[4:-4],np.array(model['grid']['x']['znr'])[4:-4]]).T
        y_grid = np.array([np.array(model['grid']['y']['znl'])[4:-4],np.array(model['grid']['y']['znc'])[4:-4],np.array(model['grid']['y']['znr'])[4:-4]]).T
        z_grid = np.array([np.array(model['grid']['z']['znl'])[4:-4],np.array(model['grid']['z']['znc'])[4:-4],np.array(model['grid']['z']['znr'])[4:-4]]).T

        x = (x_grid[:,1])[:-1]/1e5 ##en km
        x2 = (x_grid[:,1])/1e5 ##en km
        xl = np.log10(x)##radis en km i log10

        comp = 0
        for i in range(x2.shape[0]):#busca la component del radi 
            if x2[i] >= r:
                comp = i
                break

#agafa les variables entropia i vel.radial
    thd = model ['locl'] ['thd']['data']
    s = thd[:,:,:,10][4:-4,4:-4,4:-4]
    vel_r = thd[:,:,:, 5][4:-4,4:-4,4:-4]
    
#computa la compressió i l'afegeix al vector vcomp sobre totes les phi
    shocks = np.empty([0,2])
    recount = []
    for k in range(shape(z_grid)[0]):
        vcomp = []#fa el vcomp per a cada phi
        for i in range(shape(x_grid)[0], -1, -1):##do the lop goes backward
            print(i)
            if i < 299:
                dv = vel_r[k,theta,i + 1] - vel_r[k,theta,i - 1]
                vcomp.append(dv)
        vcomp = vcomp[::-1]#compresion vector

        countc = 0
        for i in np.arange(comp, 0, -1):##do the lop goes backward
            if s[k, theta, i - 1] - s[k, theta, i + 1] > 0:#s creixent
                if vcomp[i] < -epsilon*np.abs(vel_r[k, theta,i]):#compresion criterion
                    if vel_r[k, theta, i] < 0:
                                                            #if np.abs(vcomp[i]/vcomp[i+1]) > 1:
                        countc = i
                        break
        #recount.append(x2[countc])##construeix un vector amb els shocks radius
        new = np.array([z_grid[k, 1], x2[countc]]).reshape(1, -1)
        shocks = np.append(shocks, new, axis = 0)#anguloyradio
    minradt = np.array([t/10,np.amin(shocks[:,1])]).reshape(1,-1)#temps en ms i radi min en km
    minshock = np.append(minshock,minradt, axis = 0 )#temps, minimum radi
    shocksrad = np.dstack((shocksrad, shocks))#coord. de la shockwave per a cada t. El t inicial és 650ms
    timeshock = np.append(timeshock, t/10)#temps en ms
    np.save('radshock01PRE', shocksrad) #[phi, rad, step]
    np.save('minshock01PRE', minshock) #[time, min rad]
    np.save('timeshock01PRE', timeshock)#timeforeachstep





count = 0
for i in range(len(shocksrad[:,1,368])):
    if shocksrad[i,1,368] == np.amin(shocksrad[:,1,368]):
        count = i
        break

plt.plot(np.log10(x_grid[:,1]), vel_r[79,32,:]/ko)
plt.plot(np.log10(x_grid[:,1]), np.log10(s[79,32,:])/ks)
plt.legend(['Vel.radial', 'S'])
plt.xlabel('log10 R [km]')
plt.ylabel('Un.norm/ V_r [cm/s]/  np.log10 s [Kb/barion]')
plt.grid()


Vel.radial [cm/s]/  s [Kb/barion]
np.amax(np.log10(s[79,32,:]))
for i in range(len(x_grid[:,1])):
    if x_grid[i,1] == minshock[368,1]*1e5:
        count = i
        break

plt.axvline(np.log10(x_grid[124,1]), color='r', linestyle='--')

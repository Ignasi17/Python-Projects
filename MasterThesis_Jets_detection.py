######################################################################
######################################################################
                    # CaçaDolls totes magnituds #
        #A diferència del 2, este ho fa per a tots els radis posibles#
######################################################################
#AQUESTA VERSIÓ CALCULA TOTES LES MAGNITUDS (V_R, S, E_F, Q).
#CALCULA ELS JETS AMB LA MAGNITUD MARCADA AMB VARIABLE
######################################################################
                    ##ADVERTÈNCIA##
##Tot objectiu i creació d'aquest còdig pertany exclusivament a 
#Ignasi Josep Soler Poquet amb l'objectiu de desenvolupar el 
#treball de fi de màster. Qualsevol altre us agraïria que se me
#comunicara. Més que res perque crec que soles l'entenc jo.
#Gràcies.
#Igualment, si se'm vol oferir una plaça de doctorat estic disposat
#a ensenyar i permetre el seu lliure us. 

##The function of this script is the detection of jets inside the shockwave
##just after the revival of the shockwave in order to check if Soker's theory
##is right or wrong. To do that, first it computes, for spheres with fixed radii 
##for diferent radii, magnitudes such as radial velocity or energy flux. After this, 
##it recognizes the shapes and computes the barycenters...ACABAR

##aknwoledgments: To my dear friend, Salvador Mendual Sendra.

from numpy.core.fromnumeric import shape
import numpy as np
import matplotlib.pyplot as plt
import cv2
#from numpy.testing._private.utils import break_cycles

model = rdAe(11020,'/mnt/fujinas2/mobergau/sn3d/35OC/35OC/35OC-Rw/') #this first iteration is just to get the grid coordinates, which are cnstants along the time.

x_grid = np.array([np.array(model['grid']['x']['znl'])[4:-4],np.array(model['grid']['x']['znc'])[4:-4],np.array(model['grid']['x']['znr'])[4:-4]]).T
y_grid = np.array([np.array(model['grid']['y']['znl'])[4:-4],np.array(model['grid']['y']['znc'])[4:-4],np.array(model['grid']['y']['znr'])[4:-4]]).T
z_grid = np.array([np.array(model['grid']['z']['znl'])[4:-4],np.array(model['grid']['z']['znc'])[4:-4],np.array(model['grid']['z']['znr'])[4:-4]]).T

minshock = np.load('minshock.npy')

sigmas = 1.7##el nombre de sigmes escollit per al threshold per eliminar brossa

variable = 1 # 1 = v_r, 2 = energy flux density, 3 = mass flux density, 4  = entropy

dif = 4 # diferència de celles per a q siga el mateix jet


#steps = np.arange(6500, 12720,50)#12750,50

steps = np.arange(6550,6560, 10)
contornjets = []
valcoordjets = []
jets = []
timejets = []

steps = np.arange(12710,12720, 10)
for t in steps:
    model = rdAe(t,'/mnt/fujinas2/mobergau/sn3d/35OC/35OC/35OC-Rw/') #this first iteration is just to get the grid coordinates, which are cnstants along the time.

    thd = model ['locl'] ['thd']['data']


    vel_r = thd[:,:,:, 5][4:-4,4:-4,4:-4]
    vel_t = thd[:,:,:, 6][4:-4,4:-4,4:-4]
    vel_phi = thd[:,:,:, 7][4:-4,4:-4,4:-4]

    B = model['locl']['mag_vol']['data']#Magnetic field
    g = model['locl']['gravpot']['data']



    gravpot = g[:,:,:,0][4:-4,4:-4,4:-4]
    s = thd[:,:,:,10][4:-4,4:-4,4:-4]
    density = thd [ :,:,:,2][4:-4,4:-4,4:-4]#density
    P = thd[:,:,:,8][4:-4,4:-4,4:-4] #gas pressure
    U = thd[:,:,:,3][4:-4,4:-4,4:-4]#internal energy
    B_r = B[:,:,:,0][4:-4,4:-4,4:-4] ##arreglar camp amgnètic i components
    B_t = B[:,:,:,1][4:-4,4:-4,4:-4] ##arreglar camp amgnètic i components
    B_phi = B[:,:,:,2][4:-4,4:-4,4:-4] ##arreglar camp amgnètic i components

    minshock = np.load('minshock.npy')
    minrad = minshock[t/10 - 650,1]*1e5#donat que minshock comença en 650ms- Està en km el minshock.La component [:,0] són els temps totals (no postbounce). Comença en 650 ms que equival a 250 pb ASPAI
    #radi = np.linspace(0, minrad, 41)#selecciona els radis
    magnitudes = np.empty([128,64,0])
    vrmagn = np.empty([128,64,0])
    Efmagn = np.empty([128,64,0])
    EfmagnG = np.empty([128,64,0])
    Qmagn = np.empty([128,64,0])
    Smagn = np.empty([128,64,0])
    
    for i in range(shape(x_grid)[0]):#retorna comp que indica en quina component està eixe radi o el més proper
        if x_grid[i,1] >= minrad:
            it = i
            break
    
    radi = np.zeros(it + 1)
    for i in range(it + 1):
        radi[i] = x_grid[i, 1]

 # 1._ COMPUTE THE MAGNITUDES. S, V_r, Mass flux, Enegry flux

    for comp in np.arange(0, it+1, 1):
        
        r = x_grid[comp, 1]
        ##Entropía
        sr = s[:,:,comp]#agafa la entropia per al radi desitjat
        vr = np.zeros([128,64])#crea el array velocitat radial de 0

        ##Radial velocity calculus
        for j in range(shape(y_grid)[0]):#si la velocitat és major que 0, canvia la component de vr per la de vel_r. Conseguim sols les components positives de la vel.radial.
            for k in range(shape(z_grid)[0]):
                if vel_r[k , j , comp ]  >= 0:
                    vr[k, j] = vel_r[k , j , comp ]

        if variable != 0:
            vrmagn = np.dstack((vrmagn, vr))

        ###Energy fluxe density calculus.
        BE = np.zeros([128,64])
        KE = np.zeros([128,64])
        POTE = np.zeros([128,64])
        UE = np.zeros([128, 64])

        dr = (x_grid[comp,2]**3 - x_grid[comp,0]**3)*1/3
        for j in range(shape(y_grid)[0]):
            dt = np.sin(y_grid[j,1])*(y_grid[j,2] - y_grid[j,0])
            for k in range(shape(z_grid)[0]):
                rho = density[k, j, comp]
                dp = z_grid[k,2] - z_grid[k,0]
                V = dr*dt*dp
                m = rho*V
                KE[k,j] = 0.5*rho*(vel_r[k , j , comp ]**2 + vel_t[k , j , comp ]**2 + vel_phi[k, j, comp]**2)#ARRAY d'energies cièntiques
                BE[k,j] = 0.5* (B_r[k, j, comp]**2 + B_t[k , j , comp ]**2 + B_phi[k , j, comp ]**2)#se crea en la matriu BE, per a un radi donat
                UE[k,j] = U[k, j, comp]
                POTE[k, j] = rho*gravpot[k, j, comp]
        mult = (B_r[:,:,comp,]*vel_r[:,:,comp] + B_t[:,:,comp]*vel_t[:,:,comp] + B_phi[:,:,comp]*vel_phi[:,:,comp])*B_r[:,:,comp]
        Eflux = (KE + BE  + UE  + P[:,:,comp])*vel_r[:,:,comp] - mult #array Eflux- Fluxes d'energia per a cada capa ambr radi r
        Efluxg = (KE + BE  + UE + POTE + P[:,:,comp] + POTE)*vr - mult#vel_r[:,:,comp] - mult#array Eflux amb la gravitatòria

        Ef = np.zeros([128,64])
        Efg = np.zeros([128,64])
        for j in range(shape(y_grid)[0]):
            for k in range(shape(z_grid)[0]):
                if Eflux[k, j]  >= 0:
                    Ef[k,j] = Eflux[k, j]##Energy flux density array above 0
                if Efluxg[k, j] >= 0:
                    Efg[k, j] = Efluxg[k, j]
        
        

        Efmagn = np.dstack((Efmagn, Ef))
        EfmagnG = np.dstack((EfmagnG, Efg))
        ###MASS FLUX DENSITY

        Q = np.zeros([128,64])

        for j in range(shape(y_grid)[0]):
            for k in range(shape(z_grid)[0]):
                if vel_r[k , j , comp ]  >= 0:
                    Q[k,j] = density[k , j , comp ]*vel_r[k, j , comp ]##mass flux density 
        
        Qmagn = np.dstack((Qmagn, Q))
        Smagn = np.dstack((Smagn,sr))

        if variable == 1:
            trvr = vr.mean() + sigmas*np.abs(vr.std())#threshold de velocitats: <vr> + sigma
            if sigmas > 0:
                for j in range(shape(y_grid)[0]):#neteja les imatges amb el threshold
                    for k in range(shape(z_grid)[0]):
                        if vr[k,j] < trvr:
                            vr[k,j] = 0
            magnitudes = np.dstack((magnitudes, vr))
            
        if variable == 2:
            tref = Efg.mean() + sigmas*np.abs(Efg.std())#threshold de velocitats: <vr> + sigma
            if sigmas > 0:
                for j in range(shape(y_grid)[0]):#neteja les imatges amb el threshold
                    for k in range(shape(z_grid)[0]):
                        if Efg[k,j] < tref:
                            Efg[k,j] = 0
            magnitudes = np.dstack((magnitudes, Efg))
        if variable == 3:
            trq = Q.mean() + sigmas*np.abs(Q.std())
            if sigmas > 0:
                for j in range(shape(y_grid)[0]):#neteja les imatges amb el threshold
                    for k in range(shape(z_grid)[0]):
                        if Q[k, j] < trq:
                            Q[k,j] = 0
                magnitudes = np.dstack((magnitudes, Q))
        if variable == 4:
            trs = sr.mean() + sigmas*np.abs(sr.std())#threshold de velocitats: <vr> + sigma
            if sigmas > 0:
                for j in range(shape(y_grid)[0]):#neteja les imatges amb el threshold
                    for k in range(shape(z_grid)[0]):
                        if sr[k, j] < trs:
                            sr[k, j] = 0
            magnitudes = np.dstack((magnitudes, sr))

 ###############################################################################################
 ###############################################################################################
 # 2._ COMPUTE THE CONTOURS OF THE SHAPES AND THE BARYCENTERS. STORE THE OUTPUT IN 3 ARRAYS.   #
 #      BARYCENTERS-STORE THE POSITION OF THE BARYCENTERS OF EACH SHAPE                        #
 #      VALUES-STORE THE POINTS INSIDE OF EACH SHAPE AND THE VALUE OF THE MAGNITUDE            #
 #      CONTJETS-STORE THE POINTS OF THE CONTOUR OF  OF EACH SHAPE                             #
 ###############################################################################################
 ###############################################################################################
    
    barycenters = []
    coordjets = []##list que guarda les coordenades dels punts interiors de cada contorn
    contjets = []##list que guarda les coordenades del contorn dels jets
    values = []
    for contador in range(magnitudes.shape[2]):
        
        mag = magnitudes[:,:,contador]
        
        sample = np.zeros([mag.shape[0], mag.shape[1]])#matriu usada per a calcular contorns. és una còpia de mag però amb 0 i 1.

        for k in range(mag.shape[0]):
            for j in range(mag.shape[1]):
                if mag[k, j] >0:
                    sample[k,j] = 1

        res, thresh = cv2.threshold(np.uint8(sample), 0, 1, 0)[-2:]


        _,contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ##list que recolleix els baricentres per a cada imatge
        #ola = np.array([2])#vector auxiliar per fer proves i no fer tots els contours



        ##PRIMER BLOC. ACÍ AGAFE ELS CONTORNS, ELIMINE LES TAQUES MENUDES I AJUNTE ELS Q IXEN TALLATS
        kontours = np.empty([0,6])
        for i in range(len(contours)):
            kont2 = np.array(contours[i])##cada iteració és un contorn distint
            kont = np.array([kont2[:,0,1], kont2[:,0,0]]).T##transforma kont per a q siga [phi, theta]. Intercanvia les columnes de kont2 vamos
            if kont.shape[0] < 9:##primera fita per eliminar formes menudes
                continue
            
            klim = np.array([np.amax(kont[:,0]) , np.amin(kont[:,0])])#agafa els màxims i mínims de theta i phi
            jlim = np.array([np.amax(kont[:,1]) , np.amin(kont[:,1])])
        
            val = 0#selecciona el valor val que indica si està sencer o si la taca està tallada
            if klim[0] >=126 and klim[1] >1 :##val = 1 limita per baix
                val = 1
            elif klim[1] <=1 and klim[0] <126 :##val = 2 limita per dalt
                val = 2
            elif klim[0] >=126 and klim[1] <=1 :##ocupa tot
                val = 3
            else:
                val = 0

            
            bordes = np.empty([1,0])##vector que agafa els bordes de theta per a contours que delimiten
            if val == 1 or val == 2:##és a dir, si la tac és dalt o baix limitant
                for l in range(shape(kont)[0]):
                    if kont[l,0] == klim[0]:##recorre tots els punts del contorn. Si phi correspon amb el top o el bot és un punt del límit
                        bordes = np.append(bordes, kont[l,1]) 
            else:
                bordes = np.array([0])

            kontours = np.append(kontours, np.array([i,np.amin(bordes),np.amax(bordes), val, i, False ]).reshape(1,6), axis = 0)#recolleix [nº de contorn, lower limit, upper limit,val, contorn en elq  se junta, s'ajunta o s'ajunten]
            ##el vector kontours té la informació dels contorns. Indica el número en el vector contourn, els bordes i el val
        


        ##canvia els valors de kontours per a les formes que cal juntar
        for b in range(kontours.shape[0]):
            if kontours[b, 3] == 1:## si val = 1->limita per baix
                for v in range(kontours.shape[0]):
                    if kontours[v, 3] == 2:## si val =2->limita per dalt
                        if np.abs(kontours[v,1] - kontours[b,1]) <= 7 and np.abs(kontours[v,2] - kontours[b,2]) <= 7:#diferències de 7 celles o menos
                            #contours[kontours[b,0].astype(numpy.int64)] = np.concatenate((contours[kontours[b,0].astype(numpy.int64)], contours[kontours[v,0].astype(numpy.int64)]))
                            kontours[b,4] = kontours[b,0]##les formes que se coincideixen les junta i lis posa el número en q se junten
                            kontours[v,4] = kontours[b,0]
                            




        ####FA TRUE ELS QUE NO AJUNTA
        for k in range(kontours[:,0].shape[0]):
            tipo = kontours[k,0]
            for q in range(kontours[:,0].shape[0]):
                if kontours[q,4] == tipo :
                    if q == k:
                        kontours[q,5] = True
            
            #tornem a assignar els kont als nous contorns



            
        ####BLOC2. CALCUL DELS PUNTS INTERIORS
      
        bary = np.empty([0,2])
        coordlist = []
        conta0 = []
        conta0aux = -1
        for i in kontours[:,0].astype(numpy.int64):
            conta0aux = conta0aux + 1
            kont2 = np.array(contours[i])##cada iteració és un contorn distint
            kont = np.array([kont2[:,0,1], kont2[:,0,0]]).T##transforma kont per a q siga [phi, theta]. Intercanvia les columnes de kont2 vamos

            klim = np.array([np.amax(kont[:,0]) , np.amin(kont[:,0])])#agafa els màxims i mínims de theta i phi
            jlim = np.array([np.amax(kont[:,1]) , np.amin(kont[:,1])])

            #obtenim els contours i usem ara pointPolygonTest per vore quins punts estan dins del contorn.
            coord = np.empty([0,2])#vector que estarà ple dels punts dins del contorn
            for k in np.arange(klim[1], klim[0],1):
                for j in np.arange(jlim[1], jlim[0],1):
                    z = cv2.pointPolygonTest(kont, (k,j), False)
                    if z >=0:
                        coord = np.append(coord, np.array([k,j]).reshape(1,2), axis = 0)#array amb els punts dins del contorn
            
            #if coord.shape[0] < 15:##segon tall. Si és una forma molt menuda la lleve. He posat 20 a ull
            #    conta0.append(conta0aux)
            #    continue

            zeros = 0##tercer tall. Si és una figura feta per 0 majoritàriament, l'elimina    
            for k in range(coord.shape[0]):
                if mag[coord[k,0],coord[k,1]] == 0:
                            zeros = zeros + 1
            if (zeros > coord.shape[0]*1/3 or coord.shape[0] < 15) or (zeros > coord.shape[0]*1/3 and coord.shape[0] < 15):
                conta0.append(conta0aux)
                continue
            coordlist.append(coord)##estàn en l'ordre que apareixen en kontours (recordar que són els punts de dins)
        kontours = np.delete(kontours, conta0, 0)



        #######BLOC3. CÀLCUL DEL BARICENTRE
        thetatop = 0#variables que s'susen per calcular els baricentres
        phitop = 0
        bot = 0    
        zeros = 0
        magnaux2 = []
        contaux2 = []

        for i in range(len(coordlist)):

            if kontours[i,5] == False:##selecciona un contorn amb el booleà True ja uq esón als que s'ajunten o els q no estan partits
                continue
            thetatop = 0#variables que s'susen per calcular els baricentres
            phitop = 0
            bot = 0    
            zeros = 0
            contaux = np.empty([0,2])
            magnaux = np.empty([0,7])
            for j in range(len(coordlist)):##recorre els perfils en kontours per trobar els que són iguals
                if kontours[i,4] == kontours[j,4]:##selecciona sols els perfils que vagen aparellats
                    coord = coordlist[j]
                    contaux = np.concatenate([contaux, contours[kontours[j,0].astype(numpy.int64)][:,0,:]])
                    for z in range(len(coord)):##recorrec tots els punts dins de coord

                        dt = np.sin(y_grid[coord[z,1],1])*(y_grid[coord[z,1],2] - y_grid[coord[z,1],0])
                        dp = z_grid[coord[z,0],2] - z_grid[coord[z,0],0]
                        r = x_grid[comp, 1]
                        dA = r**2*dt*dp
                        thetatop = thetatop + dA*y_grid[coord[z,1],1]*mag[coord[z,0],coord[z,1]]
                        phi =  z_grid[coord[z,0],1]
                        if kontours[j,4]  !=  kontours[j,0]:##si és un anexionat s'ha mogut de dalt a baix, per tant el desplace
                            if phi <0:
                                phi = phi + 2*np.pi
                        phitop = phitop + dA*phi*mag[coord[z,0],coord[z,1]]
                        bot = dA*mag[coord[z,0],coord[z,1]] + bot 
                        magnaux = np.append(magnaux, np.array([coord[z,0], coord[z,1], magnitudes[coord[z,0],coord[z,1], contador], Qmagn[coord[z,0],coord[z,1], contador], Efmagn[coord[z,0],coord[z,1], contador], EfmagnG[coord[z,0],coord[z,1], contador] , Smagn[coord[z,0],coord[z,1], contador]] ).reshape(1,-1), axis = 0)

            contaux2.append(contaux)    
            magnaux2.append(magnaux)              
            compt = 0
            compp = 0
            for s in range(shape(y_grid)[0]):#retorna en quina component està la theta del baricentre
                if y_grid[s,1] >= thetatop/bot:
                    compt = s
                    break

            bariphi = phitop/bot
            if bariphi > z_grid[-1,1]:
                bariphi = bariphi - 2*np.pi
            for s in range(shape(z_grid)[0]):
                if z_grid[s,1]  >= bariphi :
                    compp = s
                    break
            bary = np.append(bary,np.array([compp, compt]).reshape(1,-1), axis = 0)#recolleix els baricentres com [phi, theta]
            bary
        barycenters.append(bary)
        contjets.append(contaux2)
        values.append(magnaux2)

 ###############################################################################################
 ###############################################################################################
 # 3._ Identify JETS analysing the dta from BLOC2                                              #
 #      BARYCENTERS-STORE THE POSITION OF THE BARYCENTERS OF EACH SHAPE                        #
 #      VALUES-STORE THE POINTS INSIDE OF EACH SHAPE AND THE VALUE OF THE MAGNITUDE            #
 #      CONTJETS-STORE THE POINTS OF THE CONTOUR OF  OF EACH SHAPE                             #
 ###############################################################################################
 ###############################################################################################

    baricentres = barycenters[::-1]##li done la volta per a que vaja de fora cap a dins

    tolerancies = []#és com un baricentres 2 que afegeix el nom i el booleà
    for i in range(len(baricentres)):#selecciona radi
        tol = np.empty([0,5])
        bary = baricentres[i]
        for j in range(bary.shape[0]):#selecciona baricentre en el radi
                phi = z_grid[bary[j,0],1]
                theta = y_grid[bary[j,1],1]
                tol = np.append(tol, np.array([i*10 + j , bary[j,0], bary[j,1], i*10 + j, False]).reshape(1,-1), axis = 0)
        tolerancies.append(tol)

    #range(len(tolerancies))[1:]
    for i in range(len(tolerancies))[1:]:#comença per la segona iteració.Recorre les toleràncies
        for j in range(tolerancies[i].shape[0]):#Recorre els baricentres de cada tolerància
            for k in range(tolerancies[i-1].shape[0]):##Per a cada baricentres de la tolerancia, recorre la tolerància anterior
                if tolerancies[i-1][k,1] - dif <= tolerancies[i][j,1] <= tolerancies[i-1][k,1] + dif and tolerancies[i-1][k,2] - dif <= tolerancies[i][j,2] <= tolerancies[i-1][k,2] + dif:
                    tolerancies[i][j,3] = tolerancies[i-1][k,3]
                    tolerancies[i][j,4] = True ##si pertany a un jet fa el booleà TRUE
                    tolerancies[i-1][k,4] = True


    if 100 <= minrad/1e5 < 1000:
        altura = 4
    if 1000 <= minrad/1e5 < 2000:
        altura = 4
    if 2000 <= minrad/1e5 :
        altura = 4

    jets2 = []
    #jetvalues = []
    #jetconts = []
    for i in range(len(tolerancies)):##crea el array jets. Separa en arrays distints als jets i sols els agafa si medixen més de X celles
        aux = tolerancies[i]
        for j in range(aux.shape[0]):
            if aux[j,4] == True and aux[j,0]==aux[j,3]:#si el punt és el primer dalgo pertanyent a un candidat a jet
                #puntsaux = []
                #contaux = []
                jetsaux = np.empty([0,5])
                jetsaux = np.append(jetsaux, np.array([aux[j,0], aux[j,1], aux[j,2], aux[j,3], radi[::-1][i]]).reshape(1,-1), axis = 0)
                #puntsaux.append(values[::-1][i][j])
                #contaux.append(contjets[::-1][i][j])
                for k in np.arange(i+1, len(tolerancies)):
                    aux2 = tolerancies[k]
                    for q in range(aux2.shape[0]):#agafa els punts de baix q pertanyen a  la mateixa estructura
                        if aux2[q,4] == True and aux2[q,3] == aux[j,3]:#
                            jetsaux = np.append(jetsaux, np.array([aux[j,0], aux2[q,1], aux2[q,2], aux2[q,0], radi[::-1][k]]).reshape(1,-1), axis = 0)
                    #       puntsaux.append(values[::-1][k][q])
                    #        contaux.append(contjets[::-1][k][q])
                #if jetsaux.shape[0]>4:
                if jetsaux[0,4] - jetsaux[jetsaux.shape[0]-1,4] >= minrad/altura:#sols considera jets si almenos medixen 1/3 de lamplitud total
                    jets2.append(jetsaux)#jets:[numero de jet, phi, theta, radi]
                    #jetvalues.append(puntsaux)#jetvalues(coordenades dels jets):[número de jet][radi][value,phi, theta]
                    #jetconts.append(contaux)#jetcounts(contorns dels jets):[número de jet][radi][theta, phi]

    #contornjets.append(jetconts)
    contornjets.append(contjets[::-1])    
    valcoordjets.append(values[::-1])#
    jets.append(jets2)#jets:[timestep] [numero de jet] [nom de jet, phi, theta, radi]
    timejets.append(t/10)
    data = np.array(contornjets)
    np.savez("vrcontornjetsAlt4totraditemps", data)
    data = np.array(valcoordjets)  
    np.savez("vrvalcoordjetsAlt4totraditemps", data)
    data = np.array(jets)
    np.savez("vrjetsAlt4totraditemps", data)

########################################################################################################################
######### JETSFINAL. SEPARA ELS JETS EN ENTITATS PRÒPIES I OBTÉ LA LLISTA D'INFORMACIÓ DE JETS #########################
########################################################################################################################


posicions = []
for i in range(len(jets)):#timestep
    posicionsaux = []
    for j in range(len(jets[i])):#número de jet
        #means = [''+str(timejets[i])+'+' +str(jets[i][j][0,0])+'',np.mean(jets[i][j][:,1]),np.mean(jets[i][j][:,2]),''+str(timejets[i])+'+' +str(jets[i][j][0,0])+'', False ]        
        means = [''+str(timejets[i])+'+'+str(jets[i][j][0,0])+'',np.mean(jets[i][j][:,1]),np.mean(jets[i][j][:,2]),''+str(timejets[i])+'+'+str(jets[i][j][0,0])+'', False, np.amin(jets[i][j][:,4])/1e5, np.amax(jets[i][j][:,4])/1e5,  (np.amax(jets[i][j][:,4]) -  np.amin(jets[i][j][:,4]))/1e5 ]                
        posicionsaux.append(means)
    posicions.append(posicionsaux)

# Junta els jets correlats
for i in range(len(posicions)):#step
    if len(jets[i]) == 0:
        continue
    for j in range(len(posicions[i])):#jet
        if posicions[i][j][0] == posicions[i][j][3]:#si és el primer
            bol = False
            for a in np.arange(i + 1, len(posicions),1):
                for b in range(len(posicions[a])):
                    if posicions[i][j][2] - 6 <= posicions[a][b][2] <= posicions[i][j][2] + 6 and posicions[i][j][1] - 6 <= posicions[a][b][1] <= posicions[i][j][1] + 6:
                        posicions[a][b][3] = posicions[i][j][3]
                        bol = True
            if bol == True:
                posicions[i][j][4] = True

jetsfinal = []###S'AGRUPEN ELS JETS DE TOTS ELS TEMPS.
for i in range(len(posicions)):#step
    if len(jets[i]) == 0:
        continue
    for j in range(len(posicions[i])):#jet
        if posicions[i][j][4] == True :#si és el primer
            finalaux = []
            finalaux.append(posicions[i][j])
            for a in np.arange(i + 1, len(posicions),1):
                for b in range(len(posicions[a])):
                    if posicions[a][b][3] == posicions[i][j][3]:
                        finalaux.append(posicions[a][b])
            jetsfinal.append(finalaux)


###Analyze


jetsinfo = np.empty([0,8])
jetsinfo = []
for i in range(len(jetsfinal)):
    jetstheta = []
    jetsphi = []
    for j in range(len(jetsfinal[i])):
        jetstheta.append(jetsfinal[i][j][2])
        jetsphi.append(jetsfinal[i][j][1])
    text = jetsfinal[i][-1][0]
    loc = [int(text.partition('+')[0]), float(text.partition('+')[2])]
    for t in range(len(timejets)):
        if timejets[t] == loc[0]:
            comp = t
            break
    jetsinfo.append( [jetsfinal[i][0][0], jetsfinal[i][-1][0], np.mean(jetsphi), np.mean(jetstheta),len(jetsfinal[i]), i,jetsfinal[i][-1][7], np.sum(valcoordjets[comp][int(loc[1]//10)][int(loc[1]%10)][:,4])])

jetsinfo = sorted(jetsinfo, key=lambda x: x[4])[::-1] ##Ordena els jets segons els steps






loc = [int(text.partition('+')[0]), float(text.partition('+')[2])]
comp = 0
for t in range(len(timejets)):
    if timejets[t] == loc[0]:
        comp = t
        break

##################################################################
###### LUMINOSITY ################################################
##################################################################

##Calcula la lluminositat del jet en tots els seus radis

##si vull tindre en compter l'energia gravitatòria.
t =  6840
model = rdAe(t,'/mnt/fujinas2/mobergau/sn3d/35OC/35OC/35OC-Rw/')
thd = model ['locl'] ['thd']['data']
g = model['locl']['gravpot']['data']


vel_r = thd[:,:,:, 5][4:-4,4:-4,4:-4]
density = thd [ :,:,:,2][4:-4,4:-4,4:-4]#density

gravpot = g[:,:,:,0][4:-4,4:-4,4:-4]

component = jetsinfo[0][5]
dphi = 0.049087385212340351 ##és sempre igual


Luminosity = []
LuminosityP = []
Radios = []

for a in range(len(jetsfinal[component])):##tots els instants on apareix el jet
    text = jetsfinal[component][a][0]
    loc = [int(text.partition('+')[0]), float(text.partition('+')[2])]
    compt = 0
    for t in range(len(timejets)):
        if timejets[t] == loc[0]:
            compt = t
            break
    L = []
    Lp = []
    radis = []
    Pot = []
    for i in range(len(jets[compt])):
        if jets[compt][i][0][0] == loc[1]:#Localitza el jet
            print('hola')
            print(i)
            for j in range(len(jets[compt][i])):#recorre els radis de cada jet
                text2 = jets[compt][i][j][3]#agafa el localitzador i troba el valcoordjet
                radi = jets[compt][i][j][4]
                for l in range( shape(x_grid)[0]):
                    if x_grid[l,1] >= radi:
                        comp = l
                        break
                val = valcoordjets[compt][int(text2//10)][int(text2%10)]#val és igual a les coord que pertanyen al jet
                lumi = 0
                lumip = 0

                for k in range(len(val)):#recorre tot valcoordjets del jets
                    
                    if val[k,4] > 0:
                        rho = density[val[k,0], val[k,1], comp] 
                        dt = np.sin(y_grid[val[k,1],1])*(y_grid[val[k,1],2] - y_grid[val[k,1],0])
                        potencial = density[val[k,0], val[k,1], comp]*gravpot[val[k,0], val[k,1], comp]*vel_r[val[k,0], val[k,1], comp] #*val[k,2]
        
                        lumi = radi**2*dphi*dt*(val[k][4] ) + lumi
                        lumip = radi**2*dphi*dt*(val[k][4] + potencial ) + lumip
                        #print(val[k][4])
                        #print(potencial)
                L.append(lumi)
                Lp.append(lumip)
                radis.append(radi)
            
    Luminosity.append(L)
    LuminosityP.append(Lp)
    Radios.append(radis)

for k in range(len(val)):
    if val[k,2] == 0:
        plt.plot(val[k,1], val[k,0],'r+')
       
      
        print(potencial)

for i in range(len(Radios)):
    plt.plot(Radios[i],Luminosity[i], label = ''+str(i)+'')

plt.plot(Radios[i],Luminosity[i], label = ''+str(i)+'')


plt.plot(valcoordjets[comp][int(loc[1]//10)][int(loc[1]%10)][:,1], valcoordjets[comp][int(loc[1]//10)][int(loc[1]%10)][:,0],'ro')

np.sum(valcoordjets[comp][int(loc[1]//10)][int(loc[1]%10)][:,4])

valcoordjets[comp][int(loc[1]/10)][int(loc[1]%10)][:,1]


data = np.array(jetsinfo)
np.savez("EfjetsinfoAltura4radis40", data)

vrjetsinfo[i]
Efjetsinfo[i]
Qjetsinfo[i]
Sjetsinfo[i]

###########################
##### Energy surface ######
###########################





comp = 0
for i in range(len(timejets)):
    if timejets[i] == 965:
        comp = i 
        break

comp = 0
for i in range(len(x_grid)):
    if x_grid[i,1] >= 1.63707474e+08:
        comp = i
        break

maxim = []
for i in range(len(jetsfinal)):
    print(len(jetsfinal[i]))
    maxim.append(len(jetsfinal[i]))

components = []
for i in range(len(jetsfinal)):
    if len(jetsfinal[i]) == max(maxim):
        components.append(i)

for i in range(len(valcoordjets[0])):
    val = valcoordjets[0][i]
    for j in range(len(val)):
        res = np.sum(val[j][:,5])
        print(res)
    print('--------------------------')


k = 0
lum = []
for i in range(len(jets[0][k])):
    aux = jets[0][k][i][3]
    val = valcoordjets[0][int(aux//10)][int(aux%10)]
    sum = np.sum(val[:, 5])
    lum.append(sum)
    plt.plot(jets[0][2][i][2],jets[0][2][i][1], 'ro' )


for i in range(len(jetsfinal)):
    print(len(jetsfinal[i]))


#aspect = 0.5
plt.imshow(magnitudes[:,:,124], cmap = 'coolwarm')
plt.grid()
plt.xlabel('$\\theta$')
plt.ylabel('$\phi$')
#plt.title('vel.rad.colormap, t=1018ms, r = 1358km')

for i in range(len(valcoordjets[368][0])):
    plt.plot(valcoordjets[368][0][i][:,1], valcoordjets[368][0][i][:,0],'ro')

plt.plot(contorns[368][0][2][:,0], contorns[368][0][2][:,1],'ro')

plt.plot(valcoordjets[0][0][i][:,1], valcoordjets[0][0][i][:,0],'ro')

EnGrav = []
for i in range(len(jets[0])):
    energy = []
    for j in range(len(jets[0][i])):
        text = jets[0][i][j][3]
        energy.append(np.sum(valcoordjets[0][int(text//10)][int(text%10)][:][:,4]))
    EnGrav.append(energy)

i = 135

plt.imshow(Smagn[:,:,i], cmap = 'coolwarm')
plt.xlabel('$\\theta$')
plt.ylabel('$\phi$')
plt.title('S($k_B/barion$)($r \\approx$ 2275  km)')
plt.colorbar()

plt.imshow(Qmagn[:,:,i] ,cmap = 'coolwarm')
plt.xlabel('$\\theta$')
plt.ylabel('$\phi$')
plt.title('Q($g cm^{-2}s^{-1}$)($r \\approx$ 2275 km)')
plt.colorbar()


plt.imshow(vel_r[:,:,i], cmap = 'coolwarm')
plt.xlabel('$\\theta$')
plt.ylabel('$\phi$')
plt.title('$v_r$($cm\, s^{-1}$)($r \\approx$ 3312 km)')
plt.colorbar()

plt.imshow(Efmagn[:,:,i], cmap = 'coolwarm')
plt.xlabel('$\\theta$')
plt.ylabel('$\phi$')
plt.title('$E_f$($erg\, s^{-1} \, cm^{-2}$)($r \\approx$ 3312km)')
plt.colorbar()


plt.imshow(EfmagnG[:,:,i], cmap = 'coolwarm')
plt.xlabel('$\\theta$')
plt.ylabel('$\phi$')
plt.title('$E_f$($erg\, s^{-1} \, cm^{-2}$)($r \\approx$ 2275 km)')
plt.colorbar()

for i in range(len(jetsfinal[33])):
    plt.plot(jetsfinal[33][i][2]//1, jetsfinal[33][i][2]//1,'ro' )

for i in range(len(jetsfinal[33])):
    plt.plot(i, jetsfinal[33][i][1]//1,'bo' )

for i in range(len(jets[-1][0])):
    plt.plot(i, jets[-1][0][i][2], 'bo')




 plt.imshow(sr,cmap = 'coolwarm', extent = [0,180,-180,180])
 ticks = np.arange(-180,225,45)
 plt.yticks(ticks)
 ticks = np.arange(0,225,45)
 plt.xticks(ticks)
 plt.xlabel('$\\theta$')
 plt.ylabel('$\\phi$')
 plt.grid()
 plt.colorbar()
 plt.plot(y_grid[50, 1]*180/np.pi, -z_grid[38, 1]*180/np.pi, 'ro'  )


for a in range(len(values[-1])):
    for i in range(len(values[-1][a])):
        x = y_grid[values[-1][a][i][1],1]*180/np.pi
        y = z_grid[values[-1][a][i][0],1]*180/np.pi
        plt.plot(x, -y, 'ro')

X = []
Y = []    
for a in range(cont.shape[0]):

    x = y_grid[cont[a,0],1]*180/np.pi
    y = -z_grid[cont[a,1],1]*180/np.pi  
    X.append(x)
    Y.append(y)
    plt.plot(x, -y, 'ro')  



plt.plot(y_grid[50, 1]*180/np.pi, -z_grid[38, 1]*180/np.pi, 'ro'  )
plt.plot(y_grid[x,1]*180/np.pi, -z_grid[y,1]*180/np.pi,'ro')

phi = []
theta = []
for a in range(len(jetsfinal[component])):##tots els instants on apareix el jet


    dp = 0.049087385212340351 ##és sempre igual

    phi = []
    theta = []
    text = jetsfinal[component][a][0]
    loc = [int(text.partition('+')[0]), float(text.partition('+')[2])]
    compt = 0
    for t in range(len(timejets)):
        if timejets[t] == loc[0]:
            compt = t
            break
   
    step = timejets[compt]*10
    maxphi = []
    maxtheta = []
    #model = rdAe(step,'/mnt/fujinas2/mobergau/sn3d/35OC/35OC/35OC-Rw/')
    for i in range(len(jets[compt])):
        if jets[compt][i][0][0] == loc[1]:#Localitza el jet
            for j in range(len(jets[compt][i])):#recorre els radis de cada jet
                text2 = jets[compt][i][j][3]#agafa el localitzador i troba el valcoordjet
                radi = jets[compt][i][j][4]

                for l in range( shape(x_grid)[0]):
                    if x_grid[l,1] >= radi:
                        comp = l
                        break
                val = valcoordjets[compt][int(text2//10)][int(text2%10)]#val és igual a les coord que pertanyen al jet
                maxphi.append(np.amax(val[:,0]) - np.amin(val[:,0]))
                maxtheta.append(np.amax(val[:,1]) - np.amin(val[:,1]))           

    phi.append(np.array(maxphi)*dp*180/np.pi)
    theta.append(np.array(maxtheta)*dp*180/np.pi)

for i in range(len(phi)):
    plt.plot(Radios[i], phi[i],'r--')
    plt.plot(Radios[i], theta[i],'b--')

plt.plot(Radios[-1], phi[-1],'r--',label='$\\Delta \\phi _{chorro 2}$')
plt.plot(Radios[-1], theta[-1],'b--',label='$\\Delta \\theta_{chorro 2}$')
plt.


plt.plot(Radios[-1], phi[0],'r-', label = '$\Delta \\phi _{chorro 1}$')
plt.plot(Radios[-1], theta[0], 'b-', label = '$\Delta \\theta _{chorro 1}$')
plt.plot(Radios2[-1], phi2,'r--', label = '$\Delta \\phi _{chorro 2}$')
plt.plot(Radios2[-1], theta2[0], 'b--', label = '$\Delta \\theta _{chorro 2}$')
plt.legend(loc = 'upper center')
plt.grid()
plt.xlabel('r (cm)')
plt.ylabel('$\Delta \\theta,\, \Delta \\phi\, (\circ)$')
plt.title('$\Delta \\theta\,\mathrm{y} \, \Delta \\phi\ (r)\, \mathrm{para}\, t_{chorr1} = 1271 \,\mathrm{ms} \,\mathrm{y}\, t_{chorro2} = 1102 \,\mathrm{ms}$ ')

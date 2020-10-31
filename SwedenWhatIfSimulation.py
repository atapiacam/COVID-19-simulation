# -*- coding: utf-8 -*-
"""
@author: arman
"""


import numpy as np
import matplotlib.pyplot as plt 
import random
from numba import jit
import pandas as pd

def chainProbsRandRect():
    pAcute = np.array([0.0045045045045045,
                       0.00637033136204097,
                       0.0135135135135135,
                       0.0191109940861229,
                       0.0284889879294449,
                       0.051359253382844,
                       0.0966108585113838,
                       0.141013295878365,
                       0.191109940861229])
    p1=np.full_like(pAcute,0)
    p2=np.full_like(pAcute,0)
    q1=np.full_like(pAcute,0)
    q2=np.full_like(pAcute,0)
    
    p1[0]=random.uniform(0.0,pAcute[0])
    q1[0]=np.min([pAcute[1],((pAcute[1]-pAcute[0])/(1-pAcute[0]))+((1-pAcute[1])/(1-pAcute[0]))*p1[0]])
    
    p2[0]=random.uniform(0.0,pAcute[0])
    q2[0]=np.min([pAcute[1],((pAcute[1]-pAcute[0])/(1-pAcute[0]))+((1-pAcute[1])/(1-pAcute[0]))*p2[0]])
    
    for i in range(1,8):
        p1[i]=random.uniform(p1[i-1],q1[i-1])
        q1[i]=np.max([p1[i],((pAcute[i+1]-pAcute[i])/(1-pAcute[i]))+((1-pAcute[i+1])/(1-pAcute[i]))*p1[i]])
        p2[i]=random.uniform(p2[i-1],q2[i-1])
        q2[i]=np.max([p2[i],((pAcute[i+1]-pAcute[i])/(1-pAcute[i]))+((1-pAcute[i+1])/(1-pAcute[i]))*p2[i]])
        
    p1[8]=random.uniform(p1[7],q1[7])
    p2[8]=random.uniform(p2[7],q2[7])
        
    rnd = random.uniform(0.0,1.0)
    
    pIG=p1 + rnd*(p2-p1)
    
    pSG=np.full_like(pIG,0)
    
    for i in range(0,9):
        pSG[i]=(pAcute[i]-pIG[i])/(1-pIG[i])
    
    return [pIG,pSG]


# Defines the age group to which the person will belong to 
# Adjust the probabilities depending on the demographics of your population 
@jit
def defineGroup():
    pobLimits = np.array([1186,2297,3596,4879,6150,7425,8517,9480,10000])
    num = random.randint(0, 10000)
    res = 0
    
    if num <= pobLimits[0]:
        res = 0
    elif num <= pobLimits[1]:
        res = 1
    elif num <= pobLimits[2]:
        res = 2
    elif num <= pobLimits[3]:
        res = 3
    elif num <= pobLimits[4]:
        res = 4
    elif num <= pobLimits[5]:
        res = 5
    elif num <= pobLimits[6]:
        res = 6
    elif num <= pobLimits[7]:
        res = 7
    else: 
        res = 8
        
    return res


# defines a dummy variable
@jit
def defineDummy(prob):
    
    prob = prob * 1000
    num = random.randint(0,1000) 
    
    if num <= prob:
        return 1
    else:
        return 0



# Creates a person and determines the time that he will stay in each state
@jit
def createPerson(probabilities):
    
    timeIncubated = 0
    timeAcute = 0 
    timeSymptomatic = 0
    
    ageGroup = defineGroup()
    
    
    # dummy definition process
    dumIncAcu = defineDummy(probabilities[ageGroup][0])
    
    if dumIncAcu == 1:
        dumSymAcu = -1
    else:
        dumSymAcu = defineDummy(probabilities[ageGroup][1])
    
    if(dumIncAcu == 1 or dumSymAcu == 1):
        dumAcuDead = defineDummy(probabilities[ageGroup][2])
    else:
        dumAcuDead = -1
        
    
    
    # everybody starts with an incubation period
    timeIncubated += random.triangular(5,11,7) 
    
    if(dumIncAcu == 1):
        timeAcute += random.triangular(1,14,7);
    else:
        timeSymptomatic += random.triangular(7,15,12)
        
    
    if dumSymAcu == 1:
        timeAcute += random.triangular(2,7,5)
        
    res = np.array([ageGroup, timeAcute, timeIncubated, timeSymptomatic , dumAcuDead, dumIncAcu, dumSymAcu])
    

    
    return res

# Updates the general calendar given a person 
@jit
def updateCalendar(day, person, calendar, time, peopleCreatedByGroup):
    
    ageGroup = person[0]
    timeAcute = person[1]
    timeIncubated = person[2]
    timeSymptomatic = person[3]
    seMuere = person[4]
    dumIncAcu = person[5]
    dumSymAcu = person[6]
    
    
    
    if(ageGroup == 0):
        peopleCreatedByGroup[0] += 1
    elif(ageGroup == 1):
        peopleCreatedByGroup[1] += 1
    elif(ageGroup == 2):
        peopleCreatedByGroup[2] += 1
    elif(ageGroup == 3):
        peopleCreatedByGroup[3] += 1
    elif(ageGroup == 4):
        peopleCreatedByGroup[4] += 1
    elif(ageGroup == 5):
        peopleCreatedByGroup[5] += 1
    elif(ageGroup == 6):
        peopleCreatedByGroup[6] += 1
    elif(ageGroup == 7):
        peopleCreatedByGroup[7] += 1
    else:
        peopleCreatedByGroup[8] += 1
    
    
   
    lifetime = np.full((7, time), 0)

    # the number of susceptible people reduces by one from the day the person was created till the end 
    for x in range (day,time):
        lifetime[0][x] = -1
        lifetime[6][x] = 1
       
    # one more person in incubation from the day he was created till the day he becomes symptomatic
    for x in range (day, day+timeIncubated):
        lifetime[1][x] = 1
     
        
    # Process that is executed if the person is not going to become acute from incubation 
    if dumIncAcu == 0:    

        for x in range(day + timeIncubated, day + timeIncubated + timeSymptomatic):
            lifetime[2][x] = 1
         
        if dumSymAcu == 0:
            for x in range(day + timeIncubated + timeSymptomatic, time):
                lifetime[5][x] = 1     
        else:    
            for x in range(day + timeIncubated + timeSymptomatic, day + timeIncubated + timeSymptomatic + timeAcute):
                lifetime[3][x] = 1
                if seMuere == 1:
                    for x in range(day + timeIncubated + timeSymptomatic + timeAcute, time):
                        lifetime[4][x] = 1
                else:
                    for x in range(day + timeIncubated + timeSymptomatic + timeAcute, time):
                        lifetime[5][x] = 1   
    
    # process to run if the person goes from incubation to acute 
    else:

        for x in range(day + timeIncubated, day + timeIncubated + timeAcute):
            lifetime[3][x] = 1
        
        if seMuere == 1:
            for x in range(day + timeIncubated + timeAcute, time):
                lifetime[4][x] = 1
        else:
            for x in range(day + timeIncubated + timeAcute, time):
                lifetime[5][x] = 1
    
    
    # the calendar update 
    for i in range(len(calendar)):
        for j in range(len(calendar[0])):
           calendar[i][j] = calendar[i][j] + lifetime[i][j]
           

# The simulation process
@jit
def sim(numContacts, numContacts2, numContacts3, numContacts4, numContacts5, probContagion, probContagion2, probContagion3, probContagion4, probContagion5, periodOfTime, countryPopulation):
    
    
    random.seed(30)
    
    currentDay = 1
    
    
    time = periodOfTime +50
    
    contactsPerDayIncubated = numContacts
    contactsPerDaySymptomatic = 2.5
    
    probabilityOfContagion = probContagion/10000
    
    peopleCreatedByGroup = np.array([0,0,0,0,0,0,0,0,0])
    
    totalPopulation = countryPopulation
    
      
       
    pGM = np.array([0.0222,0.0313955,0.0666,0.0941866,0.0941866,0.2531189,0.476137,0.6949699,0.9418662])
    
    pTemp=chainProbsRandRect()
    pIG=pTemp[0]
    pSG=pTemp[1]
    
    probabilities=np.vstack((pIG,pSG,pGM)).T
    
    
    
    calendar = np.full((7, time), 0) 


    for x in range (0,time):
        calendar[0][x] = totalPopulation
    

    for x in range (0,6):
        calendar[1][x] = 15
    

    for x in range (0,15):
        calendar[2][x] = 48
        
    

    while(currentDay < time-50):
        
        if (currentDay >= 21):
            contactsPerDayIncubated = numContacts2
            probabilityOfContagion = probContagion2/10000
            
        if (currentDay >= 30):
            contactsPerDayIncubated = numContacts3
            probabilityOfContagion = probContagion3/10000
        
        if (currentDay >= 50):
            contactsPerDayIncubated = numContacts4
            probabilityOfContagion = probContagion4/10000
            
        if (currentDay >= 100):
            contactsPerDayIncubated = numContacts5
            probabilityOfContagion = probContagion5/10000
            
        if (currentDay >= 177):
            contactsPerDayIncubated = 15
            probabilityOfContagion = 100/10000
            pGM = np.array([0.0222,0.0313955,0.0666,0.0941866,0.0941866,0.2531189,0.476137,0.6949699,0.9418662])/59.259
            pTemp=chainProbsRandRect()
            pIG=pTemp[0]
            pSG=pTemp[1]
            probabilities=np.vstack((pIG,pSG,pGM)).T
        
        exitRate = max(0,((calendar[1][currentDay-1]*contactsPerDayIncubated)+(calendar[2][currentDay-1]*contactsPerDaySymptomatic))*probabilityOfContagion*(calendar[0][currentDay-1]/totalPopulation))
        
        peopleCreatedToday = np.random.poisson(exitRate)
        
        for i in range(0, peopleCreatedToday):
            person = createPerson(probabilities)
            updateCalendar(currentDay,person, calendar, time,peopleCreatedByGroup)
        
       
        currentDay += 1
    
   
    return calendar


 

@jit
def main():
    
    # select the country you want to see
    # Su = Sweden 
    # No = Norway 
    # Fi = Finland 
    nordicCases = np.array(pd.read_excel('CasosNord.xlsx', index_col = 0, sheet_name = 'Su')).T
    
    
    #period of time to run the simulation
    time = 600
    
    
    # the population of the country
    countryPopulation = 10099265
    
    # This is for graphing the mean of n runs
          
    res = np.full((7, time+50), 0)     
    numberOfRuns = 1
    
    for _ in range (0,numberOfRuns):
        aux =  sim(17,15,10,9,8,137,101,101,98,105, time, countryPopulation)
        res = res + aux
    
    res = res/numberOfRuns
    

    
    accumulatedCases = res[6]

    plt.plot(res[4], label = 'Simulated')
    plt.plot(nordicCases[1].T, label = 'Offitialy reported')
    plt.xlabel('Day')
    plt.ylabel('Number of deaths')
    plt.title('Number of deaths over time')
    plt.ylim(0, 6100)
    plt.legend()
    plt.show()  

    plt.plot(accumulatedCases, label = 'Cumulative confirmed cases')
    plt.xlabel('Day')
    plt.plot(nordicCases[0].T, label = 'Offitialy reported cumulative cases')
    plt.ylabel('Simulated cumulative cases')
    plt.title('Number of symptomatic people over time')
    plt.yscale('linear')
    plt.ylim(0,5000000)
    plt.show()
    

    plt.plot(res[2])
    plt.xlabel('Day')
    plt.ylabel('Active cases')
    plt.title('Number of active cases over time')
    plt.yscale('linear')
    plt.ylim(0, 1000000)
    plt.show()
    
    plt.plot(res[5])
    plt.xlabel('Day')
    plt.ylabel('Recovered people')
    plt.title('Number of recovered people over time')
    plt.yscale('linear')
    plt.ylim(0, 5000000)
    plt.show()
 


    
if __name__ == '__main__':
    main()   























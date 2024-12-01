#villoni.py

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import chi2

print('''istruzioni:
      - per fare un fit lineare inserire come funzione "lambda x,a,b: a*x+b"
      - fit(func,x,y,sy,guess,r1,r2) #r1,r2 sono il numero di cifre decimali
      - lin_fit(x, y, sy,r1=5,r2=5,stampa=True,original_sy=0)
      - lin_fit_err(x, y, sy,r1=5,r2=5)    
      - fit_plot(func,x,y,sy,popt,title='title',ylabel='y',xlabel='x',xticks=0,yticks=0,loc='best',save=False) 
      - alfabeto(lettera) #ritorna il nome la greca
      - chi_squared(observed, expected, uncertainty, ddof,instruction=False)\n''')
    


#funzioni
def alfabeto(lettera):
    greek=[chr(code) for code in range(945,970)]
    diz = ['alpha','beta','gamma','delta','epsilon','zeta','eta','theta','iota','kappa','lambda','mu','nu','xi','omicron','pi','rho','sigma','tau','upsilon','phi','chi','psi','omega']
    if lettera in diz:
        return greek[diz.index(lettera)]
    else:
        return lettera


def print_result(p,c,r1=10,r2=10):
    alphabet = [chr(code) for code in range(97,123)]
    for i in range(len(p)):
        print(f'{alphabet[i]} : {round(p[i],r1)} +/- {round(np.sqrt(c[i][i]),r2)}')
    for i in range(len(p)):
        for j in range(i+1,len(p),1):
            print(f'cov {alphabet[i]},{alphabet[j]} : {round(c[i][j],r2)}')

def chi_squared(observed, expected, uncertainty, ddof,instruction=False,r=3):
    chi_squared = np.sum((observed - expected)**2 / uncertainty**2)
    reduced_chi_squared = chi_squared / (len(observed) - ddof)
    p_value = chi2.sf(chi_squared, len(observed) - ddof)
    alpha = 0.05

    print("Chi squared:", round(chi_squared,r-1))
    print("Reduced chi squared:", round(reduced_chi_squared,r))
    print("P value:", round(p_value,r))
    
    if (1-p_value) < alpha:
        print("Compatibilie al 95%")
    else:
        print(f"Compatibile al : {round((p_value)*100,r)}% ")

    if instruction==True:
        print("\nInstruction:")
        print("P value è l'integrale della distribuzione chi con n df da chi_squared a inf.")



def fit(func,x,y,sy,guess,r1=5,r2=5):
    popt, cov = curve_fit(func, x, y,sigma=sy,p0=guess)
    print_result(popt,cov,r1,r2)
    print('\n')
    chi_squared(y, func(x,*popt), sy, len(popt))
    return popt,cov

def fit_plot(func,x,y,sy,popt,title='title',ylabel='y',xlabel='x',xticks=0,yticks=0,loc='best',save=False):
    x_fit = np.linspace(x.min(),x.max(),1000)
    plt.figure(figsize=(10,5))
    plt.errorbar(x,y,sy,color="black",ls='', marker='.',label='dati')
    plt.plot(x_fit,func(x_fit,*popt),'-',label='fit')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if xticks!=0 and yticks!=0:
            plt.yticks(yticks)
            plt.xticks(xticks)    

    plt.grid()
    plt.legend(loc=loc)
    if save==True:
         plt.savefig(f'fit_{title}.png',dpi=300)
    plt.show()

#Funzioni per fit lineare

def my_mean(x, w):
    return np.sum( x*w ) / np.sum( w )

def my_cov(x, y, w):
    return my_mean(x*y, w) - my_mean(x, w)*my_mean(y, w)

def my_var(x, w):
    return my_cov(x, x, w)

def my_line(x, m=1, c=0):
    return m*x + c

def y_estrapolato(x, m, c, sigma_m, sigma_c, cov_mc):
    y = m*x + c
    uy = np.sqrt(np.power(x, 2)*np.power(sigma_m, 2) +
                   np.power(sigma_c, 2) + 2*x*cov_mc ) 
    return y, uy

def lin_fit(x, y, sy,r1=5,r2=5,stampa=True,original_sy=[]):
    f = lambda x,a,b: a*x+b
    #pesi
    w_y = np.power(sy.astype(float), -2) 
    
    #m
    m = my_cov(x, y, w_y) / my_var(x, w_y)
    var_m = 1 / ( my_var(x, w_y) * np.sum(w_y) )
    
    #c
    c = my_mean(y, w_y) - my_mean(x, w_y) * m
    var_c = my_mean(x*x, w_y)  / ( my_var(x, w_y) * np.sum(w_y) )
    
    #cov
    cov_mc = - my_mean(x, w_y) / ( my_var(x, w_y) * np.sum(w_y) ) 
   
    #rho
    popt=np.array([m,c])
    cov=np.array([[var_m,cov_mc],[cov_mc,var_c]]) 

    if stampa==True:
        print_result(popt,cov,r1,r2)
        print('\n')
        if len(original_sy)==0:
            chi_squared(y, f(x,*popt), sy, len(popt))
        else:
            chi_squared(y, f(x,*popt), original_sy, len(popt))
    return popt,cov

def lin_fit_err(x, y, sx,sy,r1=5,r2=5):
    popt0,_ = lin_fit(x, y, sy,stampa=False)
    new_sy = np.sqrt(sy**2 + (popt0[0]*sx)**2)
    popt,cov = lin_fit(x, y, new_sy,r1,r2,original_sy=sy)
    return popt,cov

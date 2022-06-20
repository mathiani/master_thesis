import matplotlib.pyplot as plt
import yaml
from matplotlib.collections import PolyCollection

from qdpy.algorithms import *
from qdpy.containers import *
from qdpy.plots import *
from qdpy.base import *
from qdpy import tools
import sys
import datetime
import pathlib
import shutil
import seaborn as sns
import pandas as pd
import warnings
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import gaussian_filter
warnings.filterwarnings("ignore")
import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D, art3d
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import ptitprince as pt
sns.set_theme(style="ticks", font_scale = 2)

ITERATIONS = 1000
batch_size = 1000
TITLE = "title"
fontsize = 20
y_lim = [0,1]

c2 = 'darkorange'


color = None

def plot_iterations_multiple_keys(logs, keys, save_path, save_name = "mul_plot",plot_mul = False, nr = 0):

    plt.close('all')

    plt.rcParams["figure.figsize"] = (7.5, 5.34)
    sns.set_theme(style="ticks", font_scale = 3)
    fig, ax = plt.subplots()

        
    for i in range(len(keys)):
        dashed = False
        if i == 1 or i==3:
            dashed = True
        ax = plot_iteration(logs, keys[i], save_path, mul = True, nr = i, dashed = dashed, ax = ax)

          
    ax.set_ylim(y_lim)
    #ax.tick_params(axis='y', labelcolor = 'b')
    ax.set_xlim([0,1000000])
    ax.set_ylabel('Performance',fontsize= fontsize)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:2], labels=labels[0:2], loc="upper left", fontsize=fontsize-5)
    ax.set_xlabel('evaluations',fontsize =fontsize)
   
   
    #plt.setp(ax.collections, alpha=1)
    sns.despine(offset=0, trim = True)
   

    if not nr == 0 and not nr == 2:
        ax.get_legend().remove()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    #ax.spines["left"].set_visible(True)
 

    #if nr == 1 or nr == 3:
    #    ax.spines["left"].set_visible(False)
    #    ax.axes.yaxis.set_visible(False)
        
    #plt.show()
    print(save_name)
  
 
    plt.tight_layout(pad=0.05)
    #plt.show()
    fig.savefig(save_path + save_name + ".pdf")
  
    


def plot_mul(logs, keys, save_path):

    plt.close('all')

    for i in range(len(list(logs.keys()))):
        
        
        data = {"exp_0" : logs['exp_'+ str(i)] }
        plot_iterations_multiple_keys(data, keys, save_path,save_name = "cheat_" + env + "_" + exp_names[i] + "_" + bc , plot_mul = True, nr = i)
    
  
    #plt.savefig(save_path + save_name + "_test_mul"+".pdf")
    

        
    
def plot_iteration(logs, key,save_path, mul = False, nr = 0, dashed = False, ax = None):

    
    if not mul:
        print("close")
        plt.close('all')

    nr_exp = len(logs)
    its = ITERATIONS
    largest_val = 0
    min_val = 0

    #print("NR_exp: ", nr_exp)
    
    for i in range(nr_exp):

        data = logs['exp_' +str(i)]
      
        nr_runs = len(data)
        its = ITERATIONS-1
        vals = np.zeros((nr_runs, its))
        print(key)
        for j in range(nr_runs):
            
            #print(data['run_'+ str(j)]['iterations'].keys())
            #print(data['run_'+ str(j)]['iterations'].keys())
            tmp = data['run_'+ str(j)]['iterations'][key]
            
            
            #print("list_len:", len(tmp))
            log_interval = int(len(tmp) / ITERATIONS)

          
            for k in range(its):
      
                if "[" in str(tmp[k * log_interval]) or "]" in str(tmp[k * log_interval]):
                    tmp[k * log_interval] = tmp[k * log_interval].replace("[","")
                    tmp[k * log_interval] = tmp[k * log_interval].replace("]","")


                vals[j,k] = tmp[k * log_interval]
            
            
            vals[j,:] = gaussian_filter1d(vals[j,:], sigma=3)

        if np.max(vals) > largest_val:
            largest_val = np.max(vals)

        if np.min(vals) < min_val:
            min_val = np.min(vals)
        #print(largest_val)
        df = pd.DataFrame(vals).melt()
        df.variable *= batch_size

       
        if mul:
            if nr > 1:
               
                #ax2.ylim(0,90)
                if not dashed:
                    ax = sns.lineplot(x="variable", y="value", data=df,ax = ax, color = c2, label = "Safety score")
                else:
                    ax =sns.lineplot(x="variable", y="value", data=df,ax = ax, color = c2, ls="--")

            else:

                 if not dashed:
                    ax = sns.lineplot(x="variable", y="value", data=df, ax = ax,  color = "b", label = "Fitness")
                 else:
                    ax = sns.lineplot(x="variable", y="value", data=df, ax = ax,  color = "b", ls = "--")
                
        else:
             ax = sns.lineplot(x="variable", y="value", data=df, label=exp_names[i])
            
  
             
    largest_val = largest_val*1.2
    if key == "p_valid":
        largest_val = 1
        min_val = 0


    if not mul:
        plt.savefig(save_path + "/" + key+".pdf")
        return
    #plt.show()
    return ax


def fitness_ss_kde(save_path, save_name):

    plt.rcParams["figure.figsize"] = (7.5, 5.34)
    sns.set_theme(style="ticks", font_scale = 2)

    #colors = ["blue", "darkorange", "green", "red"]
    colors = ["darkorange", "green", "red"]
    #print(sns.color_palette())

    nr_exp = len(data)
    for i in range(nr_exp):

        plt.close('all')
        #plt.rcParams["figure.figsize"] = (12,9)

        exp_data = data['exp_' + str(i)]
      
        nr_runs = len(exp_data)
        nr_inds = 0
        
        for p in range(nr_runs):
            nr_inds += len(exp_data['run_' +str(p)]['container'])
        #print(nr_inds)
            #print(len(exp_data['run_' +str(p)]['container']))

        
        vals = np.zeros((nr_inds,2))
        names = "Fitness", "Safety score"
        ind = 0
        
        for j in range(nr_runs):
            cont = exp_data['run_'+ str(j)]['container']
            for q in range(len(cont)):

                if not alg == "NEAT":
                    fit = cont[q].fitness[0]
                else:
                    fit = cont[q].fitness

                ss = cont[q].ss
                if str(fit) == "None":
                    fit = 0
                    ss = 0
                if "(" in str(fit):
                    fit = fit[0]
                

                fit = str(fit).replace("[","")
                fit = fit.replace("]","")
                ss = str(ss).replace("[","")
                ss = ss.replace("]","")

                #print(fit, ss)
                
                vals[ind,0] = fit
                vals[ind,1] = ss
                ind += 1

        q = 0
        dels = 0
        
        while q < len(vals):
            
            #print("value: ", vals[q])
            
            if vals[q][0] == 0 and vals[q][1] == 0:
                #print(vals[q])
                #print("zero: ", vals[q])
                vals = np.delete(vals,q,0)
                dels += 1
                #print("removed",vals[q])
            else:
                q += 1

        df = pd.DataFrame(data=vals,index=[i for i in range(vals.shape[0])], columns=[ names[i] for i in range(2)])
        
        ax = sns.kdeplot(data=df , x="Fitness", y="Safety score", level = 5, thresh = 0.05, color = sns.color_palette()[i+1])

        
        #plt.legend(loc="upper left")
        
        #ax.lines[0].set_color('darkorange')

        #Rocks
        plt.ylim(-325,200)
        plt.xlim(-300,325)

        #TW
        #plt.ylim(0,20)
        #plt.xlim(-40,80)
            
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        #ax.set_ylabel('Safety score',fontsize= fontsize-4)
        #ax.set_xlabel('Fitness',fontsize =fontsize-4)
        print(i)
        #if i > 0:
            #ax.spines["left"].set_visible(False)
            #ax.axes.yaxis.set_visible(False)
        plt.tight_layout(pad = 0.05)
        
        
        sns.set_theme(style="ticks", font_scale = 2)
        #sns.despine(offset=0, trim = True)
        plt.savefig(save_path + "/" + "kde_" + env + "_" + exp_names[i] + "_" + bc + ".pdf")
        #plt.show()

    
def plot_bar(logs, key, save_path,swarm = False):

    
    fitness = False
    ss = True
    
    plt.close('all')
    plt.rcParams["figure.figsize"] = (15, 8)

    data = logs
    nr_exp = len(data)

    
    vals1 = list()
    vals2 = list()
    vals3 = list()
    vals4 = list()
    vals5 = list()
    vals6 = list()
    vals7 = list()
    for i in range(nr_exp):
        print("i:", i)
        exp_data = data['exp_' + str(i)]
        nr_runs = len(exp_data)
        ##print(nr_runs)
       
        
        for j in range(nr_runs):

            cont = exp_data['run_'+ str(j)]['container']
            print(len(cont))
            for ind in cont:

                if ss:

                    if i == 0:
                        vals1.append(ind.ss)
                    if i == 1:
                        vals2.append(ind.ss)
                    if i == 2:
                        vals3.append(ind.ss)
                    if i == 3:
                        vals4.append(ind.ss)
                    if i == 4:
                        vals5.append(ind.ss)
                    if i == 5:
                        vals6.append(ind.ss)
                    if i == 6:
                        vals7.append(ind.ss)
                if fitness and not ss:
                  
                    if not isinstance(ind.fitness,float):
                        ind.fitness = ind.fitness[0]

                    if i == 0:
                        vals1.append(ind.fitness)
                    if i == 1:
                        vals2.append(ind.fitness)
                    if i == 2:
                        vals3.append(ind.fitness)
                    if i == 3:
                        vals4.append(ind.fitness)
                    if i == 4:
                        vals5.append(ind.fitness)
                    if i == 5:
                        vals6.append(ind.fitness)
                    if i == 6:
                        vals7.append(ind.fitness)

    print("------------------")
    #vals1.extend(vals1)
    #vals3.extend(vals3)
    vals3.extend(vals3)
    #vals5.extend(vals5)
    #vals5.extend(vals5)
    
    data = [vals1,vals2,vals3, vals4, vals5,vals6,vals7]

    max_len = max(len(data[0]),len(data[1]),len(data[2]),len(data[3]),len(data[4]),len(data[5]),len(data[6]))
    for i in range(len(data)):
        print(len(data[i]))
        
        for j in range(max_len - len(data[i])):
            data[i].append(np.nan)
    a = np.zeros(len(data[0]))
    a[:] = np.NaN


    #With NEAT
    #df = pd.DataFrame({"NEAT:":a,
    #                   "NS:Task Specific": data[1],
    #                   "NS:Generic": data[2],
    #                   "NSLC:Task Specific": data[3],
    #                   "NSLC:Generic": data[4],
    #                   "MAP-Elites:Task Specific": data[5],
    #                   "MAP-Elites:Generic": data[6]})

    #Without NEAT
    df = pd.DataFrame({
                       "NS:Task Specific": data[1],
                       "NS:Generic": data[2],
                       "NSLC:Task Specific": data[3],
                       "NSLC:Generic": data[4],
                       "ME:Task Specific": data[5],
                       "ME:Generic": data[6]})

    
    df = df.melt()
    df["Algorithm"] = df["variable"].apply(lambda x: x.split(":")[0])
    df["BC"] = df["variable"].apply(lambda x: x.split(":")[1])
    df = df.rename(columns={'value': 'Safety score'})

    df2 = pd.DataFrame({"NEAT:":data[0],
                       "NS:Task Specific": a,
                       "NS:Generic": a,
                       "NSLC:Task Specific": a,
                       "NSLC:Generic": a,
                       "MAP-Elites:Task Specific": a,
                       "MAP-Elites:Generic": a})
    df2 = df2.melt()
    df2["Algorithm"] = df2["variable"].apply(lambda x: x.split(":")[0])
    df2 = df2.rename(columns={'value': 'Safety score'})
    
   
  
    sns.set_theme(style="white", font_scale = 2.7)

    dx = "Safety score"
    dy = "Algorithm"
    dhue = "BC"
    ort = "h"
    pal = "Set2"
    sigma = .2
    fig, ax = plt.subplots()


    if swarm:
        ax = sns.swarmplot(data=df2,x = "Algorithm", y="Safety score", ax=ax,size=2.5)
        
        ax = sns.swarmplot(data=df, x="Algorithm", y="Safety score", dodge = True, hue = "BC", palette = ['b','darkorange'], ax=ax,size=2)
    else:

        #sns.set(style="whitegrid", font_scale=2.7)
        #ax = sns.violinplot(data=df2,x = "Algorithm", y="Safety score", ax=ax,size=2.5)

        #ax = pt.RainCloud(x=dy, y=dx, hue=dhue, data=df, palette=pal, bw=sigma,
                          #width_viol=.7, ax=ax, orient=ort)
        #ax = pt.RainCloud(data=df, x="Algorithm", y="Safety score", ax=ax, hue = "BC", palette= "Set2", orient="h",
        #        width_viol = .6, jitter=1.0, move= 0.2, point_size=2, bw = 0.2)

        ax = pt.half_violinplot(data=df, y="Algorithm", x="Safety score", ax=ax,split = True, hue = "BC", palette= "Set2",bw = .2,
                      scale = "area", width = .6, inner = None, alpha = 0.7, orient="h")
        #plt.setp(ax.collections, alpha=.3)
        ax = sns.stripplot(data=df, y="Algorithm", x="Safety score", ax=ax, hue = "BC", palette= "Set2",size = 2, jitter = 1, zorder = 0)

    #Add significance indicator    

    #x1, x2 = -0.2, 0.2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    #y, h, col = 16.1, 0.5, 'k'
    #plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

    #x1, x2 = 1.8, 2.2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    #y, h, col = 16.1, 0.5, 'k'
    #plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

    #x1, x2 = 0.8, 1.2   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    #y, h, col = 16.1, 0.5, 'k'
    #plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    #plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
    
    
    #plt.setp(ax.collections, alpha=1)
    sns.despine(offset=-10, trim = True)
    ax.get_legend().remove()
    ax.set_xlabel('Safety Score')
    ax.set_ylabel('')
    ax.spines["left"].set_visible(False)
    #ax.spines["bottom"].set_visible(False)
    
    #plt.legend(bbox_to_anchor=(0.17, 0.98), loc='upper right', borderaxespad=0,fontsize = 30)
    plt.tight_layout(pad=0.05)
    #plt.show()
    print(save_path + ".pdf")
    plt.savefig(save_path + ".pdf")


def setAlpha(ax, a):
    for art in ax.get_children():
        if isinstance(art, PolyCollection):
            art.set_alpha(a)
 
def plot_3D():

    k = 1
    shape = (30*k,30*k)
    a = np.zeros((30*k,30*k))

    a[9*k,9*k] = 1.4*k
    a[10*k,20*k] = 1*k
    a[20 *k,20*k] = -1.2*k
    a[21*k,9*k] = 0.7*k
    

    a = gaussian_filter(a, sigma=3)
    
    lin_x = np.linspace(0,200,shape[0],endpoint=False)
    lin_y = np.linspace(0,200,shape[1],endpoint=False)
    x,y = np.meshgrid(lin_x,lin_y)
    plt.rcParams["figure.figsize"] = (6, 5)
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_surface(x,y,a,cmap='jet', edgecolor='black', antialiased=False,linewidth=0.4)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_zlim(-0.01,0.05)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    #plt.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel("Fitness",rotation=90)
    ax.zaxis.label.set_rotation(90)

    #for ii in xrange(0,360,1):
    ax.view_init(elev=14., azim=46)
    plt.savefig("Fitness_landscape.pdf")
    
    plt.show()

    
def plot_3D_QD():
    #plt.rcParams['axes.facecolor'] = 'none'
    k = 2
    shape = (30*k,30*k)
    a = np.zeros((30*k,30*k))

    a[9*k,9*k] = 0.8*k
    a[16*k,23*k] = 1.4*k
    a[23 *k,16*k] = 1.8*k
    #a[21*k,9*k] = 0.7*k
    

    a = gaussian_filter(a, sigma=5)
    
    lin_x = np.linspace(0,30*k,shape[0],endpoint=False)
    lin_y = np.linspace(0,30*k,shape[1],endpoint=False)
    x,y = np.meshgrid(lin_x,lin_y)
    plt.rcParams["figure.figsize"] = (6, 5)
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111, projection="3d")
    
    ax.plot_surface(x,y,a, color = 'w', edgecolor='black', antialiased=False,linewidth=0.4, alpha = 1, zorder=-0)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_zlim(0,0.05)
    ax.axes.yaxis.set_visible(False)
    ax.axes.xaxis.set_visible(False)
    #plt.axis('off')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    #ax.set_xlabel("Dimension 1")
    #ax.set_ylabel("Dimension 2")
    ax.zaxis.set_rotate_label(False) 
    ax.set_zlabel("Fitness",rotation=90)
    ax.zaxis.label.set_rotation(90)

    #for ii in xrange(0,360,1):
    #ax.view_init(elev=14., azim=46)
    #plt.savefig("Fitness_landscpae.pdf")

    
    
    #for i in range(7):
        #ax.scatter3D(6+i, 9, a[6+i,9]+0.001,marker='o',s = 100, cmap = 'jet',zorder=0)

    #ax.scatter3D(16*k, 23*k, a[16*k,23*k],marker='o',s = 100, cmap = 'jet',zorder=0,c='r')
    #ax.scatter3D(23*k, 16*k, a[23*k,16*k],marker='o',s = 100, cmap = 'jet',zorder=0,c='r')
    plt.show()

def stats():

    vals = {}

    #print(nr_exp)
    
    for i in range(nr_exp):
         
        vals[i] = list()
        exp_data = data['exp_' + str(i)]
        nr_runs = len(exp_data)

        
        for j in range(nr_runs):
            #print(exp_data['run_'+str(j)]['iterations']['max_ss'][1])
            vals[i].append(float(exp_data['run_'+str(j)]['iterations']['max_ss'][len(exp_data['run_'+str(j)]['iterations']['max_ss'])-1]))

    print(len(vals[3]))
    #vals[3] = vals[3] * int((len(vals[4])/len(vals[3])))
    #vals[5] = vals[5] * int((len(vals[6])/len(vals[5])))

    
    print(vals[1])
    print(vals[2])

    print(vals[3])
    print(vals[4])

    print(vals[5])
    print(vals[6])
            
    stats1, p1 = mannwhitneyu(vals[1]*2,vals[2]*2)
    print(stats1, p1)
    
    stats2, p2 = mannwhitneyu(vals[3]*2,vals[4])
    print(stats2, p2)

    stats3, p3 = mannwhitneyu(vals[5]*2,vals[6])
    print(stats3, p3)

    a = multipletests([p1,p2,p3],alpha = 0.01,method="holm")
    print(a)
    #print(vals[1])
    #print(np.median(vals[1]))
    print("Mdn:", np.median(vals[1]), np.median(vals[2])    ," U:",stats1    ,"p:" ,round(a[1][0],5))

    print("Mdn:", np.median(vals[3]), np.median(vals[4])    ," U:",stats2    ,"p:" ,round(a[1][1],5))

    print("Mdn:", np.median(vals[5]), np.median(vals[6])    ," U:",stats2    ,"p:" ,round(a[1][2],5))


def plot_proof_of_consept():

    env_dict = { "rd": "/home/mathias/real_experiments/rocks_diamonds/new_ts/", "tw": "/home/mathias/real_experiments/tomato_watering/new_ts/"}
    envs = ["rd", "tw"]
    algorithms = ["NS", "NSLC", "ME"]
    env_max_fitness = {"rd": 93,"tw": 14}
    data = {}

    for env in envs:
        path = env_dict[env]
        data[env] = dict()
        for alg in algorithms:

            data[env][alg] = {"nr_evals": list(), "highest_fitness": list(), }
            alg_path = path + alg
            for i in range(5):
                with open(alg_path + f"/run_{i}/final.p", 'rb') as f:
                    print("Opening:", alg_path + f"run_{i}/final.p")
                    container= pickle.load(f)["container"]


                sorted_inds = sorted(container, key= lambda x: x.fitness, reverse=True)
                found = False
                nr_evals = 0
                #print(len(sorted_inds))
                for ind in sorted_inds:
                    nr_evals += 1
                    if ind.fitness[0] == ind.ss:
                        data[env][alg]["nr_evals"].append(nr_evals)
                        data[env][alg]["highest_fitness"].append(ind.fitness[0]/env_max_fitness[env])
                        found = True
                        break

                if not found:
                    #If none found
                    data[alg]["nr_evals"].append(len(sorted_inds))
                    data[alg]["highest_fitness"].append(0)
    '''
    df = pd.DataFrame({
        "NS:Nr_evals": data["NS"]["nr_evals"],
        "NS:Highest_fitness": data["NS"]["highest_fitness"],
        "NSLC:Nr_evals": data["NSLC"]["nr_evals"],
        "NSLC:Highest_fitness": data["NSLC"]["highest_fitness"],
        "ME:Nr_evals": data["ME"]["nr_evals"],
        "ME:Highest_fitness": data["ME"]["highest_fitness"]
    })
    '''
    #df_fit = pd.DataFrame({
    #    "NS": data["NS"]["highest_fitness"],
    #    "NSLC": data["NSLC"]["highest_fitness"],
    #    "ME": data["ME"]["highest_fitness"]
    #})
    #df_fit = df_fit.melt()
    #df_fit["Algorithm"] = df_fit["variable"].apply(lambda x: x.split(":")[0])
    #df_fit = df_fit.rename(columns={'value': 'Fitness'})



    df_evals = pd.DataFrame({
        "NS:Rocks and Diamonds": data["rd"]["NS"]["nr_evals"],
        "NSLC:Rocks and Diamonds": data["rd"]["NSLC"]["nr_evals"],
        "ME:Rocks and Diamonds": data["rd"]["ME"]["nr_evals"],
        "NS:Tomato Watering": data["tw"]["NS"]["nr_evals"],
        "NSLC:Tomato Watering": data["tw"]["NSLC"]["nr_evals"],
        "ME:Tomato Watering": data["tw"]["ME"]["nr_evals"]
    })
    df_evals = df_evals.melt()
    df_evals["Algorithm"] = df_evals["variable"].apply(lambda x: x.split(":")[0])
    df_evals = df_evals.rename(columns={'value': 'Number of evaluations until safe individual'})
    df_evals["Environment"] = df_evals["variable"].apply(lambda x: x.split(":")[1])

    df_fit = pd.DataFrame({
        "NS:Rocks and Diamonds": data["rd"]["NS"]["highest_fitness"],
        "NSLC:Rocks and Diamonds": data["rd"]["NSLC"]["highest_fitness"],
        "ME:Rocks and Diamonds": data["rd"]["ME"]["highest_fitness"],
        "NS:Tomato Watering": data["tw"]["NS"]["highest_fitness"],
        "NSLC:Tomato Watering": data["tw"]["NSLC"]["highest_fitness"],
        "ME:Tomato Watering": data["tw"]["ME"]["highest_fitness"]
    })
    df_fit = df_fit.melt()
    df_fit["Algorithm"] = df_fit["variable"].apply(lambda x: x.split(":")[0])
    df_fit = df_fit.rename(columns={'value': 'Highest safe fitness'})
    df_fit["Environment"] = df_fit["variable"].apply(lambda x: x.split(":")[1])


    fig, ax = plt.subplots()
    sns.set_style("white")
    box1 = sns.boxplot(data = df_evals, y= "Number of evaluations until safe individual", x="Environment", hue = "Algorithm",ax = ax, palette = ["darkorange", "green","red"])
    plt.show()

    fig, ax = plt.subplots()
    box1 = sns.boxplot(data = df_fit, y= 'Highest safe fitness', x="Environment", hue = "Algorithm",ax = ax, palette = ["darkorange", "green","red"])

    plt.show()


    '''
    boxcolors = ["orange", "red", "green"]
    for patch in box1['boxes']:
        #patch.set(color="red")
        patch.set(facecolor=boxcolors[i])
        i+=1
    i = 0
    for patch in box2['boxes']:
        #patch.set(color="red")
        patch.set(facecolor=boxcolors[i])
        i += 1
        pass
    '''

    ##################################################################################################################################################
plot_proof_of_consept()

nr_exp = int((len(sys.argv)-1)/2)
data = {}

exp_names = list()
print(nr_exp)
for i in range(nr_exp):
    exp_names.append(sys.argv[nr_exp+1+i])

save_path = sys.argv[len(sys.argv)-1]

print(save_path)

for i in range(nr_exp):        
        with open(sys.argv[i+1], 'rb') as f:
            data["exp_"+str(i)] = pickle.load(f)

bc = "None"

if "task_spesific".upper() in sys.argv[1].upper() or "ts" in sys.argv[1] :
    bc = "ts"
elif "medium".upper() in sys.argv[1].upper():
    bc= "medium_bc"
if "generic".upper() in sys.argv[1].upper():
    bc= "gn"


alg = "None"

if "ME" in sys.argv[1]:
    alg = "ME"
elif "NEAT".upper() in sys.argv[1].upper():
    alg= "NEAT"
elif "NSLC".upper() in sys.argv[1].upper():
    alg = "NSLC"
elif "NS".upper() in sys.argv[1].upper():
    alg = "NS"

env = "None"

if "rocks_diamonds".upper() in sys.argv[1].upper():
    env = "rd"
elif "tomato_watering".upper() in sys.argv[1].upper():
    env = "tw"


#print(data['exp_0']['run_0']['iterations'])

#Lets plot!

#Usage python3 compare_exp.py -log-files - exp-names -save dir 

#Example:  "python3 compare_exp.py /home/mathias/real_experiments/rocks_diamonds/new_ts/ME/logs.pkl /home/mathias/real_experiments/rocks_diamonds/new_ts/NS/logs.pkl /home/mathias/real_experiments/rocks_diamonds/new_ts/NSLC/logs.pkl /home/mathias/real_experiments/rocks_diamonds/new_ts/ME/logs.pkl NEAT NS NSLC MAP-ELITES /home/mathias/real_experiments/rocks_diamonds/new_ts/"




#Can plot: P_valid, max (fitness), avg (fitness), max_ss, mean_ss, plot_3D() (grid), final_pop_plot, qd_score(ss),  

#fitness_ss_kde(save_path, save_name = env + "_" + alg + "_" +  bc)

#stats()
#plot_bar(data, "max_ss", save_path,swarm = False)


#plot_mul(data,["avg","max", "mean_ss", "max_ss"], save_path)
#plot_mul(data,["p_valid"], save_path)

#plot_3D()
#plot_3D_QD()



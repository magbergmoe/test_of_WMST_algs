#######################################################
#######################################################
#######################################################
# This piece of code can be used to create a visual
# represenatation of the testing results produced by
# 'testing.py'
#
# After running 'testing.py', you should have the following
# folder structure:
# 
# Folder
#  |-- testing.py
#  |-- creating_plots.py
#  |-- results
#  |      |-- normalWeights_normalPreds_independent
#  |      |      |-- test1.csv
#  |      |      |-- test1.txt
#  |      |      |-- test2.csv
#  |      |      |-- test2.txt
#  |      |      ...
#  |      |      |-- test10000.csv
#  |      |      |-- test10000.txt
#  |      |-- normalWeights_normalPreds_not_indepndent
#  |      |      |-- test1.csv
#  |      |      |-- test1.txt
#  |      |      |-- test2.csv
#  |      |      |-- test2.txt
#  |      |      ...
#  |      |      |-- test10000.csv
#  |      |      |-- test10000.txt
#  |      |-- normalWeights_unifPreds
#  |      |      |-- test1.csv
#  |      |      |-- test1.txt
#  |      |      |-- test2.csv
#  |      |      |-- test2.txt
#  |      |      ...
#  |      |      |-- test10000.csv
#  |      |      |-- test10000.txt
#  |      |-- unifWeights_normalPreds_independent
#  |      |      |-- test1.csv
#  |      |      |-- test1.txt
#  |      |      |-- test2.csv
#  |      |      |-- test2.txt
#  |      |      ...
#  |      |      |-- test10000.csv
#  |      |      |-- test10000.txt
#  |      |-- unifWeights_normalPreds_not_indepndent
#  |      |      |-- test1.csv
#  |      |      |-- test1.txt
#  |      |      |-- test2.csv
#  |      |      |-- test2.txt
#  |      |      ...
#  |      |      |-- test10000.csv
#  |      |      |-- test10000.txt
#  |      |-- unifWeights_unifPreds
#  |      |      |-- test1.csv
#  |      |      |-- test1.txt
#  |      |      |-- test2.csv
#  |      |      |-- test2.txt
#  |      |      ...
#  |      |      |-- test10000.csv
#  |      |      |-- test10000.txt
#
#
# Based on this folder structure, this scipt runs through each 
# type of test (corresponding to subfolders), and then it reads the results
# of the tests based on the contents of the '.txt' files, that 
# contain the most important information about the tests.
# 
# For each '.txt' file, this script creates three points on an (x,y)-plot.
# On point for the algorithm FtP, one for GFtP, and one for MarkingAlg.
# The values on the first axis is the normalized error of the instance,
# and the value on the second axis is the formance ratio ALG(I)/OPT(I)
# on the instance I.
# In the final plot, the results for FtP are marked in red, the results 
# for GFtP are marked in green, and the results for MarkingAlg are marked in blue.
#
# The resulting plots are saved in a folder called 'plots', on the same layer as 'results'


import matplotlib.pyplot as plt
import os


for folder in os.listdir('results/'):
    x = []
    y_ftp = []
    y_gftp = []
    y_markingalg = []
    count = 0
    if not folder.endswith('Store'):
        for file in sorted(os.listdir('results/' + folder)):
            if file.endswith('.txt'):
                lines = open('results/' + folder + '/' + file).readlines()
                epsilon = float(lines[6].split(': ')[1].removesuffix('\n'))
                c_opt = float(lines[7].split(': ')[1].removesuffix('\n'))
                c_ftp = float(lines[8].split(': ')[1].removesuffix('\n'))
                c_gftp = float(lines[9].split(': ')[1].removesuffix('\n'))
                c_markingalg = float(lines[10].split(': ')[1].removesuffix('\n'))

                x.append(epsilon)
                y_ftp.append(c_ftp/c_opt)
                y_gftp.append(c_gftp/c_opt)
                y_markingalg.append(c_markingalg/c_opt)
                count += 1

        print('Number of samples included:',count)

        fig, axs = plt.subplots(2,1,figsize=(20,13),gridspec_kw={'height_ratios': [7, 1]})

        plt.suptitle('\u03B5/Performance-ratio plot on 10000 randomly generated graphs\n using the Erdös-Rényi-Gilbert model', fontsize = 20,fontweight = 'bold', wrap = True)

        axs[0].plot(x,y_ftp,'o',color='red', label = 'FtP')
        axs[0].plot(x,y_gftp,'o',color = 'green', label = 'GFtP')
        axs[0].plot(x,y_markingalg,'o',color ='blue', label = 'Third algorithm')

        match folder:
            case 'normalWeights_normalPreds_independent':
                axs[0].set_title('w: N(0.5,0.15),     w\u0302: N(0.5,0.15)',fontsize = 18)
            case 'normalWeights_normalPreds_not_indepndent':
                axs[0].set_title('w: N(0.5,0.15),     w\u0302: N(w,0.15)',fontsize = 18)
            case 'normalWeights_unifPreds':
                axs[0].set_title('w: N(0.5,0.15),     w\u0302: U(0,1)',fontsize = 18)
            case 'unifWeights_normalPreds_independent':
                axs[0].set_title('w: from U(0,1),     w\u0302: N(0.5,0.15)',fontsize = 18)
            case 'unifWeights_normalPreds_not_indepndent':
                axs[0].set_title('w: from U(0,1),     w\u0302: N(w,0.15)',fontsize = 18)
            case 'unifWeights_unifPreds':
                axs[0].set_title('w: from U(0,1),     w\u0302: U(0,1)',fontsize = 18)
        
        
        axs[0].set_xlabel('\u03B5', fontsize = 18)
        axs[1].set_xlabel('\u03B5', fontsize = 18)
        axs[0].set_ylabel('Performance ratio', fontsize = 18)
        legend = axs[0].legend(fontsize = 18)

        axs[1].boxplot(x,vert=False)

        axs[0].grid(True)
        axs[1].grid(True)

        fig.savefig('plots/' + folder + '.png')

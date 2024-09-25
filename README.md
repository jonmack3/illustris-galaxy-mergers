# illustris-galaxy-mergers
Code and sample files used in the analysis of "On the Frequency of Multiple Galaxy Mergers in Cosmological Simulations"

Set configuration parameters in config.py, then run multiple_mergers.py. The functions in multiple_mergers.py are designed to run in numerical order per config.py: run setmergers (or setmergersmult to run multiple instances of setmergers sequentially), then setmassmax, setfs, setdtmax, setdts, and finally setfmin. (Numerical data is generated in .npz format; use viewmergerdata, viewfvmdata, or viewdtdata to see text versions of the numerical output.) Any of the plotting functions may then be run. The code is designed to run by Illustris number and z, with all values of all other configuration parameters, i.e. when running setfs, run it at Illustris-1 and z = 5 and all values of other parameters (mu 2, 4, and 10, etc), for each combination of Illustris run number and z. Sample numerical output files with the correct configuration parameters are in output/numerical. Running each plotting function independently (as opposed to through createpubplots) will cause plots to be generated with titles and configuration parameter figtext.
